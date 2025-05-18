# stream_client.py (modified example)
import cv2
import requests
import json
import base64
import numpy as np
import time
import os
from pathlib import Path 

API_URL = "http://127.0.0.1:8000/process_g2_frame/"
VIDEO_SOURCE = 1 # or "path/to/video.mp4"
OUTPUT_JSON_DIR = "./video_analysis_output" # Define your output directory
VIDEO_ID = "game_session_01" # An identifier for the current video

cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_count = 0

# Create the base output directory if it doesn't exist
if not os.path.exists(OUTPUT_JSON_DIR):
    os.makedirs(OUTPUT_JSON_DIR)
    print(f"Created output directory: {OUTPUT_JSON_DIR}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if isinstance(VIDEO_SOURCE, str): # If it's a file, reset or break
            print("End of video file.")
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            # continue
            break 
        else: # Webcam error
            print("Failed to grab frame from webcam.")
            break


    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {'image_file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    # Define the polygon for the full frame
    h, w = frame.shape[:2]
    field_polygon = [[0,0],[w-1,0],[w-1,h-1],[0,h-1]]

    data = {
        'playing_field_polygon_coords_json': json.dumps(field_polygon),
        'return_annotated_image': 'true', # Set to 'false' if you only want JSON data
        'save_to_json_dir': OUTPUT_JSON_DIR,
        'video_filename': VIDEO_ID,
        'frame_number': str(frame_count) # Send frame_count as a string
    }

    try:
        response = requests.post(API_URL, files=files, data=data, timeout=10) # Increased timeout
        if response.status_code == 200:
            result = response.json()
            if result.get("annotated_image_base64"):
                annotated_bytes = base64.b64decode(result["annotated_image_base64"])
                np_arr = np.frombuffer(annotated_bytes, np.uint8)
                annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv2.imshow("Annotated Frame", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # else:
            #     print(f"Frame {frame_count}: JSON data processed (no annotated image returned).")

        else:
            print(f"⚠️ Frame {frame_count}: Server error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Frame {frame_count}: Request failed: {e}")
        # Optionally, add a short delay and retry, or break
        time.sleep(1) # Wait a bit before trying again or exiting
        # break 

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Processing finished. JSON data (if enabled) saved under {OUTPUT_JSON_DIR}/{VIDEO_ID}")