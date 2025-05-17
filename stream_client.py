import cv2
import requests
import json
import base64
import numpy as np

import time

API_URL = "http://localhost:8000/process_frame/"
VIDEO_SOURCE = 1  # Use 0 for webcam, or "path/to/video.mp4"

cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {'image_file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {
        'playing_field_polygon_coords_json': json.dumps([[0,0],[frame.shape[1]-1,0],[frame.shape[1]-1,frame.shape[0]-1],[0,frame.shape[0]-1]]),
        'return_annotated_image': 'true'
    }

    try:
        response = requests.post(API_URL, files=files, data=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get("annotated_image_base64"):
                annotated_bytes = base64.b64decode(result["annotated_image_base64"])
                np_arr = np.frombuffer(annotated_bytes, np.uint8)
                annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv2.imshow("Annotated Frame", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print(f"⚠️ Server error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print("🚨 Request failed:", e)
        break

cap.release()
cv2.destroyAllWindows()
