# initial_processing_client.py
import cv2
import requests
import json
import base64
import numpy as np
import time
import os
from pathlib import Path

API_URL = "http://localhost:8000/process_frame/"

# --- Configuration ---
RAW_VIDEO_CLIPS_DIR = "./videos/nonPasses" # Folder containing clip1.mp4, clip2.mp4 etc.
TEMP_JSON_OUTPUT_DIR = "./temp_json_output2" # Where initial JSONs will be saved
# --- End Configuration ---

Path(TEMP_JSON_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

video_files = [f for f in os.listdir(RAW_VIDEO_CLIPS_DIR) if f.lower().endswith(('.mp4', '.mkv', '.avi'))]

for video_filename_original in video_files:
    video_source_path = os.path.join(RAW_VIDEO_CLIPS_DIR, video_filename_original)
    
    # Use the original video filename (sanitized) as the 'video_filename' for the API
    # This will create subfolders like temp_json_output/clip1_mp4/frame_000.json
    api_video_filename = Path(video_filename_original).stem # "clip1" from "clip1.mp4"
    
    print(f"\nProcessing raw clip: {video_source_path}")
    print(f"  API save_to_json_dir: {TEMP_JSON_OUTPUT_DIR}")
    print(f"  API video_filename: {api_video_filename}")

    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_source_path}")
        continue

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, img_encoded = cv2.imencode(".jpg", frame)
        files = {'image_file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        h, w = frame.shape[:2]
        field_polygon = [[0,0],[w-1,0],[w-1,h-1],[0,h-1]]

        data_for_api = {
            'playing_field_polygon_coords_json': json.dumps(field_polygon),
            'return_annotated_image': 'true', # Keep 'true' for this step for visual inspection
            'save_to_json_dir': TEMP_JSON_OUTPUT_DIR,
            'video_filename': api_video_filename,
            'frame_number': str(frame_count)
        }

        try:
            response = requests.post(API_URL, files=files, data=data_for_api, timeout=15)
            if response.status_code == 200:
                result = response.json()
                if result.get("annotated_image_base64"): # Display for verification
                    annotated_bytes = base64.b64decode(result["annotated_image_base64"])
                    np_arr = np.frombuffer(annotated_bytes, np.uint8)
                    annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    cv2.imshow(f"Annotated - {api_video_filename}", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): # Allow 'q' to quit processing this clip
                        print(f"  User quit processing for {api_video_filename}")
                        cap.release() # Release current video capture
                        break 
                if frame_count % 30 == 0:
                    print(f"  Frame {frame_count} processed for {api_video_filename}.")
            else:
                print(f"  ⚠️ Frame {frame_count} ({api_video_filename}): Server error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"  🚨 Frame {frame_count} ({api_video_filename}): Request failed: {e}")
            time.sleep(1)
        frame_count += 1
    
    cap.release()
    cv2.destroyWindow(f"Annotated - {api_video_filename}") # Close specific window
    print(f"  Finished initial processing for {video_source_path}.")

cv2.destroyAllWindows() # Close any remaining OpenCV windows
print("\nInitial processing of all raw video clips complete.")
print(f"JSON outputs are in subfolders within: {TEMP_JSON_OUTPUT_DIR}")