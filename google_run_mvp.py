import cv2
import numpy as np
from ultralytics import YOLO
import subprocess # For listing cameras on Linux
import uuid # For generating truly unique IDs if needed, though tracker IDs are usually sufficient

# --- COCO Keypoints ---
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
BODY_PART_MAP = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "upper_body": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_leg": ["left_hip", "left_knee", "left_ankle"],
    "right_leg": ["right_hip", "right_knee", "right_ankle"],
    "left_foot": ["left_ankle"],
    "right_foot": ["right_ankle"]
}
SKELETON_CONNECTIONS = [
    (KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("left_shoulder")),
    (KEYPOINT_NAMES.index("right_hip"), KEYPOINT_NAMES.index("right_shoulder")),
    # ... (rest of skeleton connections) ...
    (KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("right_shoulder")),
    (KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("right_hip")),
    (KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("left_elbow")),
    (KEYPOINT_NAMES.index("left_elbow"), KEYPOINT_NAMES.index("left_wrist")),
    (KEYPOINT_NAMES.index("right_shoulder"), KEYPOINT_NAMES.index("right_elbow")),
    (KEYPOINT_NAMES.index("right_elbow"), KEYPOINT_NAMES.index("right_wrist")),
    (KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("left_knee")),
    (KEYPOINT_NAMES.index("left_knee"), KEYPOINT_NAMES.index("left_ankle")),
    (KEYPOINT_NAMES.index("right_hip"), KEYPOINT_NAMES.index("right_knee")),
    (KEYPOINT_NAMES.index("right_knee"), KEYPOINT_NAMES.index("right_ankle")),
    (KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("nose")),
    (KEYPOINT_NAMES.index("right_shoulder"), KEYPOINT_NAMES.index("nose")),
    (KEYPOINT_NAMES.index("nose"), KEYPOINT_NAMES.index("left_eye")),
    (KEYPOINT_NAMES.index("nose"), KEYPOINT_NAMES.index("right_eye")),
    (KEYPOINT_NAMES.index("left_eye"), KEYPOINT_NAMES.index("left_ear")),
    (KEYPOINT_NAMES.index("right_eye"), KEYPOINT_NAMES.index("right_ear")),
]

PLAYING_FIELD_POLYGON = None

# --- Helper Functions (largely the same) ---
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_distance_sq(p1, p2):
    if p1 is None or p2 is None: return float('inf')
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def is_inside_box(point, box):
    if point is None or box is None: return False
    x, y = point; x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_inside_polygon(point, polygon):
    if point is None: return False
    if polygon is None: return True
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

def get_keypoints_dict(keypoints_tensor, conf_tensor):
    kpts = {}
    if keypoints_tensor is None or conf_tensor is None: return kpts
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(keypoints_tensor) and i < len(conf_tensor):
            x, y = keypoints_tensor[i]; conf = conf_tensor[i]
            if conf > 0.3: kpts[name] = {'xy': (int(x), int(y)), 'conf': conf.item()}
    return kpts

# --- Load Models ---
print("Loading models...")
try:
    # For person tracking, we'll use the object_model with tracking enabled
    object_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt") # Pose model typically doesn't do tracking itself
except Exception as e:
    print(f"Error loading YOLO models: {e}"); exit()
print("Models loaded.")

# --- Open Webcam ---
CAMERA_INDEX = 2 # Adjusted back to 0 for common default
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Failed to open webcam with index {CAMERA_INDEX}.") # ... (error handling) ...
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam opened successfully: {frame_width}x{frame_height}")

if PLAYING_FIELD_POLYGON is None:
    print(f"PLAYING_FIELD_POLYGON not set. Using full webcam frame.")
    PLAYING_FIELD_POLYGON = np.array([[0,0],[frame_width-1,0],[frame_width-1,frame_height-1],[0,frame_height-1]], np.int32)

# --- Global storage for tracked player data ---
# This will store data per tracked_player_id (UUID)
# Example: {1: {'last_seen_frame': 100, 'performance_log': [], 'color': (R,G,B)}, ...}
tracked_players_data = {}
next_player_uuid_int = 1 # Simple integer ID from tracker
player_colors = {} # To assign a consistent color to each player ID

def get_player_color(player_id):
    if player_id not in player_colors:
        # Generate a new color for new players
        player_colors[player_id] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
    return player_colors[player_id]

# -----------------------
# Main Loop
# -----------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    debug_print = (frame_count % 30 == 0)
    if debug_print: print(f"\n--- Processing frame {frame_count} ---")

    if PLAYING_FIELD_POLYGON is not None:
        cv2.polylines(frame, [PLAYING_FIELD_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

    # --- Object Detection and Tracking (for Persons) ---
    # Use a tracker config file. Ultralytics provides some by default.
    # e.g., 'botsort.yaml' or 'bytetrack.yaml'
    # Ensure these files are in your working directory or provide a full path
    # You might need to copy them from the ultralytics library or create a simple one.
    # A simple bytetrack.yaml might just be empty or contain default tracker args.
    # For simplicity, we'll try to use the default if available by string name.
    try:
        # Tracking specific classes (person: 0 for COCO)
        tracked_object_results = object_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)[0]
    except Exception as e:
        print(f"Error during tracking: {e}. Ensure tracker config (e.g., bytetrack.yaml) is valid/accessible.")
        # Fallback to simple detection if tracking fails
        tracked_object_results = object_model(frame, classes=[0], verbose=False)[0]


    ball_info = None
    current_frame_tracked_persons = [] # Store {'box': ..., 'center': ..., 'tracker_id': ...}

    if tracked_object_results.boxes is not None and tracked_object_results.boxes.id is not None:
        for box_data, track_id_tensor in zip(tracked_object_results.boxes, tracked_object_results.boxes.id):
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            center = get_center((x1, y1, x2, y2))
            conf = box_data.conf.item()
            # cls_id = int(box_data.cls) # Already filtered by classes=[0] for person
            tracker_id = int(track_id_tensor.item()) # Get the integer ID from the tensor

            if not is_inside_polygon(center, PLAYING_FIELD_POLYGON):
                continue

            current_frame_tracked_persons.append({
                'box': (x1, y1, x2, y2),
                'center': center,
                'tracker_id': tracker_id, # This is our persistent ID
                'conf': conf
            })

            # Update tracked_players_data
            if tracker_id not in tracked_players_data:
                tracked_players_data[tracker_id] = {
                    'first_seen_frame': frame_count,
                    'performance_log': [], # List to store performance metrics
                    'color': get_player_color(tracker_id)
                }
            tracked_players_data[tracker_id]['last_seen_frame'] = frame_count
            tracked_players_data[tracker_id]['current_box'] = (x1, y1, x2, y2) # Store current box for pose association

            # Draw person box with tracker ID
            player_color = tracked_players_data[tracker_id]['color']
            cv2.rectangle(frame, (x1, y1), (x2, y2), player_color, 2)
            cv2.putText(frame, f"P-ID: {tracker_id} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, player_color, 2)

    # --- Ball Detection (no tracking needed for ball, simple detection is fine) ---
    ball_detections = object_model(frame, classes=[32], verbose=False)[0] # COCO class for sports ball is 32
    if ball_detections.boxes is not None:
        for det in ball_detections.boxes:
            x1b, y1b, x2b, y2b = map(int, det.xyxy[0])
            center_b = get_center((x1b, y1b, x2b, y2b))
            conf_b = det.conf.item()
            if is_inside_polygon(center_b, PLAYING_FIELD_POLYGON) and conf_b > 0.3:
                if ball_info is None or conf_b > ball_info['conf']:
                    ball_info = {'box': (x1b, y1b, x2b, y2b), 'center': center_b, 'conf': conf_b}
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 165, 255), 2)
                cv2.putText(frame, f"ball {conf_b:.2f}", (x1b, y1b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)


    # --- Pose Estimation (on the whole frame) ---
    pose_results_list = pose_model(frame, verbose=False)[0]
    all_poses_in_frame = []
    if pose_results_list.keypoints is not None and \
       hasattr(pose_results_list.keypoints, 'xy') and pose_results_list.keypoints.xy is not None and \
       hasattr(pose_results_list.keypoints, 'conf') and pose_results_list.keypoints.conf is not None:
        num_poses = len(pose_results_list.keypoints.xy)
        if num_poses > 0 and num_poses == len(pose_results_list.keypoints.conf):
            for i in range(num_poses):
                kpts_tensor = pose_results_list.keypoints.xy[i]
                conf_tensor = pose_results_list.keypoints.conf[i]
                keypoints_dict = get_keypoints_dict(kpts_tensor, conf_tensor)
                if keypoints_dict:
                    # Check if pose is roughly within field using nose or hip
                    core_kpt_name = "nose" if "nose" in keypoints_dict else ("left_hip" if "left_hip" in keypoints_dict else None)
                    if core_kpt_name and keypoints_dict[core_kpt_name].get('xy') is not None and \
                       is_inside_polygon(keypoints_dict[core_kpt_name]['xy'], PLAYING_FIELD_POLYGON):
                        all_poses_in_frame.append(keypoints_dict)

    # --- Associate Poses with Tracked Persons ---
    players_with_pose_and_id = [] # List of dicts: {'tracker_id':..., 'box':..., 'keypoints':...}

    for person_info in current_frame_tracked_persons:
        person_box = person_info['box']
        person_center = person_info['center']
        tracker_id = person_info['tracker_id']
        best_match_pose_kpts = None
        min_dist_pose = float('inf')

        for pose_kpts in all_poses_in_frame: # Iterate over all poses detected in the frame
            # Use a central keypoint of the pose for association
            ref_kpt_name = "nose" if "nose" in pose_kpts else ("left_hip" if "left_hip" in pose_kpts else None)
            if ref_kpt_name and pose_kpts[ref_kpt_name].get('xy') is not None:
                kpt_xy = pose_kpts[ref_kpt_name]['xy']
                if is_inside_box(kpt_xy, person_box):
                    dist = get_distance_sq(kpt_xy, person_center)
                    if dist < min_dist_pose:
                        min_dist_pose = dist
                        best_match_pose_kpts = pose_kpts
        
        if best_match_pose_kpts:
            players_with_pose_and_id.append({
                'tracker_id': tracker_id,
                'box': person_box,
                'keypoints': best_match_pose_kpts,
                'color': tracked_players_data[tracker_id]['color'] # Get color from global store
            })
            # Store the pose in the global player data for this frame (optional)
            tracked_players_data[tracker_id]['current_pose'] = best_match_pose_kpts


    if debug_print:
        print(f"  Tracked Persons: {len(current_frame_tracked_persons)}")
        print(f"  Poses in Frame: {len(all_poses_in_frame)}")
        print(f"  Players with Pose and ID: {len(players_with_pose_and_id)}")

    # --- Draw Player Body Parts and Skeletons (for players with pose and ID) ---
    for player_data in players_with_pose_and_id:
        kpts = player_data['keypoints']
        player_color = player_data['color'] # Use the assigned consistent color

        # Skeleton drawing and body part labeling (same as before, but use player_color)
        for part_name, kp_names_in_part in BODY_PART_MAP.items():
            avg_pos_coords = []
            for kp_name in kp_names_in_part:
                if kp_name in kpts and kpts[kp_name].get('xy') is not None:
                    pt = kpts[kp_name]['xy']
                    if part_name in ["left_foot", "right_foot", "head"]: avg_pos_coords.append(pt)
            if avg_pos_coords:
                center_of_part = (int(np.mean([p[0] for p in avg_pos_coords])), int(np.mean([p[1] for p in avg_pos_coords])))
                if part_name in ["left_foot", "right_foot"]:
                     cv2.putText(frame, part_name, (center_of_part[0], center_of_part[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, player_color, 1)
                     cv2.circle(frame, center_of_part, 5, player_color, -1)
                elif part_name == "head" and "nose" in kpts and kpts["nose"].get('xy') is not None:
                    cv2.putText(frame, "Head", (kpts["nose"]['xy'][0], kpts["nose"]['xy'][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, player_color, 1)
        for i_skel, j_skel in SKELETON_CONNECTIONS:
            kp_name1, kp_name2 = KEYPOINT_NAMES[i_skel], KEYPOINT_NAMES[j_skel]
            if kp_name1 in kpts and kp_name2 in kpts and \
               kpts[kp_name1].get('xy') is not None and kpts[kp_name2].get('xy') is not None:
                pt1 = kpts[kp_name1]['xy']; pt2 = kpts[kp_name2]['xy']
                cv2.line(frame, pt1, pt2, player_color, 1) # Use player_color for skeleton

    # --- Possession Detection (using players_with_pose_and_id) ---
    possessor_tracker_id = None # Store tracker_id of possessor
    min_ball_dist_sq = float('inf')
    POSSESSION_THRESHOLD_SQ = 4500 # Increased threshold slightly

    if ball_info and ball_info.get('center') is not None:
        ball_center = ball_info['center']
        for player_data in players_with_pose_and_id:
            kpts = player_data['keypoints']
            relevant_kpt_names = ["left_ankle", "right_ankle", "left_knee", "right_knee"]
            closest_player_part_dist_sq = float('inf')
            for kp_name in relevant_kpt_names:
                if kp_name in kpts and kpts[kp_name].get('xy') is not None:
                    part_xy = kpts[kp_name]['xy']
                    dist_sq = get_distance_sq(part_xy, ball_center)
                    if dist_sq < closest_player_part_dist_sq: closest_player_part_dist_sq = dist_sq
            if closest_player_part_dist_sq < POSSESSION_THRESHOLD_SQ:
                if closest_player_part_dist_sq < min_ball_dist_sq:
                    min_ball_dist_sq = closest_player_part_dist_sq
                    possessor_tracker_id = player_data['tracker_id']
                    possessor_box = player_data['box'] # Get box of current possessor

    if possessor_tracker_id is not None:
        # Log possession event (example)
        if 'possession_frames' not in tracked_players_data[possessor_tracker_id]:
            tracked_players_data[possessor_tracker_id]['possession_frames'] = 0
        tracked_players_data[possessor_tracker_id]['possession_frames'] += 1
        
        # Draw possession indicator
        if 'current_box' in tracked_players_data[possessor_tracker_id]: # Use the latest box
             px1, py1, px2, py2 = tracked_players_data[possessor_tracker_id]['current_box']
             cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
             cv2.putText(frame, f"P-ID {possessor_tracker_id} In Possession", (px1, py1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # --- Performance Logging Example ---
    # For each player with pose, you can log metrics
    for p_data in players_with_pose_and_id:
        tid = p_data['tracker_id']
        # Example: Log current keypoints or derived angles
        # tracked_players_data[tid]['performance_log'].append(
        #     {'frame': frame_count, 'keypoints': p_data['keypoints']}
        # )
        # If you implement squat analysis or other exercise analysis:
        # squat_score = calculate_squat_score(p_data['keypoints'])
        # if squat_score:
        #    tracked_players_data[tid]['performance_log'].append({'frame': frame_count, 'squat_score': squat_score})
        pass # Placeholder for actual performance metric logging

    # --- Show Output ---
    cv2.imshow("YOLO Sports Analysis with Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- After the loop, print summary of tracked player data (example) ---
print("\n--- Tracked Player Summary ---")
for player_id, data in tracked_players_data.items():
    print(f"Player ID (Tracker): {player_id}")
    print(f"  First seen: frame {data['first_seen_frame']}")
    print(f"  Last seen: frame {data['last_seen_frame']}")
    if 'possession_frames' in data:
        print(f"  Frames in possession: {data['possession_frames']}")
    # print(f"  Performance log entries: {len(data['performance_log'])}")
    # You could save data['performance_log'] to a file (JSON, CSV) here.
