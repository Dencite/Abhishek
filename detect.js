import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import path from 'node:path';

const MODEL_PATH = path.join(process.cwd(), 'best.onnx');
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.70;
const NMS_IOU_THRESHOLD = 0.45;

const classNames = [
  '10_New',   // 0
  '10_Old',   // 1
  '20',       // 2
  '50_New',   // 3
  '50_Old',   // 4
  '100_New',  // 5
  '100_Old',  // 6
  '200',      // 7
  '500'       // 8
];

let sessionPromise;

function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['cpu']
    }).catch((error) => {
      sessionPromise = undefined;
      throw error;
    });
  }
  return sessionPromise;
}

function preprocessRgb(rgbBuffer) {
  const pixels = INPUT_SIZE * INPUT_SIZE;
  const float32Data = new Float32Array(3 * pixels);

  for (let i = 0; i < pixels; i++) {
    float32Data[i] = rgbBuffer[i * 3] / 255.0;
    float32Data[i + pixels] = rgbBuffer[i * 3 + 1] / 255.0;
    float32Data[i + 2 * pixels] = rgbBuffer[i * 3 + 2] / 255.0;
  }

  return new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function calculateIoU(a, b) {
  const xA = Math.max(a.x, b.x);
  const yA = Math.max(a.y, b.y);
  const xB = Math.min(a.x + a.w, b.x + b.w);
  const yB = Math.min(a.y + a.h, b.y + b.h);

  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const unionArea = a.w * a.h + b.w * b.h - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

function applyNMS(boxes) {
  const remaining = [...boxes].sort((a, b) => b.score - a.score);
  const selected = [];

  while (remaining.length) {
    const current = remaining.shift();
    selected.push(current);

    for (let i = remaining.length - 1; i >= 0; i--) {
      if (calculateIoU(current, remaining[i]) >= NMS_IOU_THRESHOLD) {
        remaining.splice(i, 1);
      }
    }
  }

  return selected;
}

function processOutput(output, imgWidth, imgHeight) {
  const data = output.data;
  const dims = output.dims;

  // This project currently uses the YOLO output shape [1, 13, 8400]:
  // 4 box values + 9 class scores, matching the original browser code.
  const numClasses = classNames.length;
  const numChannels = 4 + numClasses;
  const numAnchors = 8400;

  if (!dims || dims.length !== 3 || dims[1] !== numChannels || dims[2] !== numAnchors) {
    throw new Error(`Unexpected model output shape: ${JSON.stringify(dims)}. Expected [1, 13, 8400].`);
  }

  const boxes = [];
  const plane = numAnchors;

  for (let i = 0; i < numAnchors; i++) {
    let maxScore = -Infinity;
    let classId = -1;

    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * plane + i];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore < CONF_THRESHOLD) continue;

    const cx = data[i];
    const cy = data[plane + i];
    const w = data[2 * plane + i];
    const h = data[3 * plane + i];

    const x = ((cx - w / 2) / INPUT_SIZE) * imgWidth;
    const y = ((cy - h / 2) / INPUT_SIZE) * imgHeight;
    const boxW = (w / INPUT_SIZE) * imgWidth;
    const boxH = (h / INPUT_SIZE) * imgHeight;
    const aspectRatio = boxH !== 0 ? boxW / boxH : 0;

    // Preserve the filtering used by the original web app.
    if (boxW > 60 && boxH > 40 && (aspectRatio >= 1.2 || aspectRatio <= 0.83)) {
      boxes.push({
        x,
        y,
        w: boxW,
        h: boxH,
        classId,
        className: classNames[classId],
        score: maxScore
      });
    }
  }

  return applyNMS(boxes);
}

function jsonResponse(body, status = 200) {
  return Response.json(body, {
    status,
    headers: {
      'Cache-Control': 'no-store',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  });
}

export async function OPTIONS() {
  return jsonResponse({ ok: true });
}

export async function GET() {
  return jsonResponse({
    ok: true,
    endpoint: '/api/detect',
    method: 'POST',
    message: 'Send a JPEG image as the raw request body with Content-Type: image/jpeg.'
  });
}

export async function POST(request) {
  try {
    const contentType = request.headers.get('content-type') || '';
    if (!contentType.toLowerCase().includes('image/jpeg') &&
        !contentType.toLowerCase().includes('image/jpg')) {
      return jsonResponse({
        ok: false,
        error: 'Send the image as raw JPEG bytes with Content-Type: image/jpeg.'
      }, 415);
    }

    const input = Buffer.from(await request.arrayBuffer());

    if (!input.length) {
      return jsonResponse({ ok: false, error: 'Empty image body.' }, 400);
    }

    // Decode JPEG, resize exactly like the browser app, and expose RGB bytes.
    const { data: rgb, info } = await sharp(input)
      .rotate()
      .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'fill' })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const session = await getSession();
    const inputName = session.inputNames[0] || 'images';
    const tensor = preprocessRgb(rgb);
    const results = await session.run({ [inputName]: tensor });
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];

    if (!outputTensor) {
      throw new Error(`Model did not return output '${outputName}'.`);
    }

    const detections = processOutput(outputTensor, info.width, info.height);

    return jsonResponse({
      ok: true,
      detected: detections.length ? detections[0].className : null,
      confidence: detections.length ? detections[0].score : 0,
      count: detections.length,
      detections: detections.map((d) => ({
        className: d.className,
        confidence: Number(d.score.toFixed(4)),
        box: {
          x: Number(d.x.toFixed(1)),
          y: Number(d.y.toFixed(1)),
          width: Number(d.w.toFixed(1)),
          height: Number(d.h.toFixed(1))
        }
      }))
    });
  } catch (error) {
    console.error('Detection error:', error);
    return jsonResponse({
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    }, 500);
  }
}
