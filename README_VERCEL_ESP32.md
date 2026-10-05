# Vercel + ESP32-CAM Currency Detection

This version keeps the original browser-side detector and adds a server endpoint:

`POST https://onnx-currency-detector.vercel.app/api/detect`

The endpoint accepts a raw JPEG request body and returns JSON containing the detected note and confidence.

## Deploy

1. Put these files in the GitHub repository used by your Vercel project:
   - `index.html`
   - `script.js`
   - `style.css`
   - `best.onnx`
   - `package.json`
   - `vercel.json`
   - `api/detect.js`

2. Push the changes to GitHub.

3. In Vercel, open the project and redeploy the latest commit. Vercel automatically discovers functions under `/api` and installs dependencies from `package.json`.

4. After deployment, open:
   `https://onnx-currency-detector.vercel.app/api/detect`
   A successful GET should return JSON saying the endpoint is ready for POST.

5. Test with a JPEG before connecting the ESP32-CAM:

```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary "@currency.jpg" \
  https://onnx-currency-detector.vercel.app/api/detect
```

Expected shape:

```json
{
  "ok": true,
  "detected": "100_New",
  "confidence": 0.94,
  "count": 1,
  "detections": []
}
```

## ESP32-CAM

Open `ESP32_CAM_Vercel_Send.ino` in Arduino IDE / your ESP32 environment.

Change:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_OR_PHONE_HOTSPOT";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";
```

The serial monitor trigger is:

`1` + Enter -> capture JPEG -> POST to Vercel -> print JSON response.

## Important

Keep the JPEG payload comfortably below Vercel's 4.5 MB Function request limit. QVGA (320x240) with JPEG quality around 12 is a sensible starting point for the ESP32-CAM demo.

This is a cloud inference path, not 30-FPS inference. Each capture requires Wi-Fi upload + server inference + response.
