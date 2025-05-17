import cv2
import numpy as np
from ultralytics import YOLO
import subprocess # For listing cameras on Linux
import cv2.aruco as aruco # Import ArUco library

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

# --- Helper Functions ---
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_aruco_center(corners):
    # corners is a 1x4x2 array from ArUco detection
    cX = int(np.mean(corners[0,:,0]))
    cY = int(np.mean(corners[0,:,1]))
    return (cX, cY)

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
    object_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    print("Please ensure you have 'yolov8n.pt' and 'yolov8n-pose.pt'")
    print("You can typically download them by running a YOLO command once or checking Ultralytics docs.")
    exit()
print("Models loaded.")

# --- ArUco Setup ---
print("Initializing ArUco detector...")
ARUCO_DICT_NAME = "DICT_4X4_100" # Example, choose one that matches your markers
aruco_dict = None
try:
    # For OpenCV versions >= 4.7.0 (and some earlier 4.x versions)
    # The direct attribute access is common for predefined dictionaries
    aruco_dictionary_id_cv_attr = getattr(cv2.aruco, ARUCO_DICT_NAME)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dictionary_id_cv_attr)
except AttributeError:
    # Fallback or alternative method, often for older OpenCV or if the above fails
    print(f"AttributeError with direct access to {ARUCO_DICT_NAME}, trying Dictionary_get method...")
    try:
        # Map string name to integer value for Dictionary_get
        # This mapping might need to be more extensive depending on the dictionaries you use
        if ARUCO_DICT_NAME == "DICT_4X4_50":
            aruco_dictionary_id_val = cv2.aruco.DICT_4X4_50
        elif ARUCO_DICT_NAME == "DICT_4X4_100":
            aruco_dictionary_id_val = cv2.aruco.DICT_4X4_100
        elif ARUCO_DICT_NAME == "DICT_4X4_250":
            aruco_dictionary_id_val = cv2.aruco.DICT_4X4_250
        elif ARUCO_DICT_NAME == "DICT_4X4_1000":
            aruco_dictionary_id_val = cv2.aruco.DICT_4X4_1000
        elif ARUCO_DICT_NAME == "DICT_5X5_50":
            aruco_dictionary_id_val = cv2.aruco.DICT_5X5_50
        # ... Add more common dictionaries
        elif ARUCO_DICT_NAME == "DICT_6X6_250": # A common one
            aruco_dictionary_id_val = cv2.aruco.DICT_6X6_250
        else:
            print(f"ArUco dictionary name {ARUCO_DICT_NAME} not recognized in fallback. Using DICT_4X4_100 by default.")
            aruco_dictionary_id_val = cv2.aruco.DICT_4X4_100 # Default fallback
        
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dictionary_id_val)
    except Exception as e_aruco_dict:
        print(f"Could not initialize ArUco dictionary {ARUCO_DICT_NAME}: {e_aruco_dict}")
        print("Ensure your OpenCV and opencv-contrib-python versions are compatible and support ArUco.")
        print("Common dictionaries: DICT_4X4_50, DICT_4X4_100, DICT_6X6_250, etc.")
        exit()

if aruco_dict is None:
    print("Failed to load any ArUco dictionary. Exiting.")
    exit()

# CORRECTED LINE:
aruco_parameters = aruco.DetectorParameters() # Use this for newer OpenCV versions
# If DetectorParameters() does not work and you get an error saying it's not callable or similar,
# and you are on a very old OpenCV (e.g. 3.x), you might need:
# aruco_parameters = aruco.DetectorParameters_create()
# But DetectorParameters() is the standard for OpenCV 4.x+

print("ArUco detector initialized.")


# --- Open Webcam ---
CAMERA_INDEX = 2 # Set your camera index
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Failed to open webcam with index {CAMERA_INDEX}.")
    try:
        print("\nTrying to list available cameras (Linux only, needs v4l-utils):")
        result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True, check=True)
        print(result.stdout)
    except FileNotFoundError:
        print("  'v4l2-ctl' not found. Install 'v4l-utils' to list cameras on Linux.")
    except subprocess.CalledProcessError as e:
        print(f"  Error listing cameras: {e}")
    except Exception as e_gen:
        print(f"  Could not list cameras: {e_gen}")
    print("\nEnsure your webcam is connected and not used by another application.")
    print("If you have multiple cameras, try changing CAMERA_INDEX (e.g., to 1, 2).")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam opened successfully: {frame_width}x{frame_height}")

if PLAYING_FIELD_POLYGON is None:
    print(f"PLAYING_FIELD_POLYGON not set. Using full webcam frame.")
    PLAYING_FIELD_POLYGON = np.array([[0,0],[frame_width-1,0],[frame_width-1,frame_height-1],[0,frame_height-1]], np.int32)

# --- Global storage for tracked player data ---
tracked_players_data = {} 
player_colors = {} 

def get_player_color(player_id):
    if player_id not in player_colors:
        player_colors[player_id] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
    return player_colors[player_id]

# -----------------------
# Main Loop
# -----------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame from webcam (stream end?). Exiting.")
        break

    frame_count += 1
    debug_print = (frame_count % 30 == 0)
    if debug_print: print(f"\n--- Processing frame {frame_count} ---")

    if PLAYING_FIELD_POLYGON is not None:
        cv2.polylines(frame, [PLAYING_FIELD_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- ArUco Marker Detection ---
    corners, ids, rejected_img_points = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    
    detected_aruco_players = {}
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])
            marker_corners = corners[i]
            x_coords = marker_corners[0,:,0]
            y_coords = marker_corners[0,:,1]
            aruco_x1, aruco_x2 = int(min(x_coords)), int(max(x_coords))
            aruco_y1, aruco_y2 = int(min(y_coords)), int(max(y_coords))
            aruco_box = (aruco_x1, aruco_y1, aruco_x2, aruco_y2)
            aruco_center = get_aruco_center(marker_corners)
            detected_aruco_players[marker_id] = {
                'box_aruco': aruco_box,
                'center_aruco': aruco_center,
                'corners': marker_corners
            }

    # --- YOLO Object Detection and Tracking (for Persons) ---
    try:
        # It's good practice to have a bytetrack.yaml file in your directory or ensure ultralytics can find it.
        # If it's missing, tracking might default to something else or raise an error depending on ultralytics version.
        yolo_tracked_results = object_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)[0]
    except Exception as e:
        if debug_print: print(f"Warning: Error during YOLO tracking: {e}. Using simple detection for persons.")
        yolo_tracked_results = object_model(frame, classes=[0], verbose=False)[0]

    current_frame_persons_yolo = []
    if yolo_tracked_results.boxes is not None:
        yolo_ids_present = hasattr(yolo_tracked_results.boxes, 'id') and yolo_tracked_results.boxes.id is not None
        for i in range(len(yolo_tracked_results.boxes)):
            box_data = yolo_tracked_results.boxes[i]
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            center = get_center((x1, y1, x2, y2))
            conf = box_data.conf.item()
            yolo_tracker_id = None
            # Check if 'id' attribute exists and is not None before trying to access it
            if yolo_ids_present and yolo_tracked_results.boxes.id is not None and i < len(yolo_tracked_results.boxes.id):
                 yolo_tracker_id_tensor = yolo_tracked_results.boxes.id[i]
                 if yolo_tracker_id_tensor is not None: # Ensure the tensor itself is not None
                    yolo_tracker_id = int(yolo_tracker_id_tensor.item())

            if not is_inside_polygon(center, PLAYING_FIELD_POLYGON): continue
            current_frame_persons_yolo.append({
                'box_person': (x1, y1, x2, y2),
                'center_person': center,
                'yolo_tracker_id': yolo_tracker_id,
                'conf': conf
            })

    # --- Consolidate Player Information ---
    consolidated_players_this_frame = []
    for aruco_id, aruco_data in detected_aruco_players.items():
        player_id_str = f"A-{aruco_id}"
        best_yolo_match_box = aruco_data['box_aruco']
        min_dist_yolo = float('inf')
        for yolo_person in current_frame_persons_yolo:
            if is_inside_box(aruco_data['center_aruco'], yolo_person['box_person']):
                dist = get_distance_sq(aruco_data['center_aruco'], yolo_person['center_person'])
                if dist < min_dist_yolo:
                    min_dist_yolo = dist
                    best_yolo_match_box = yolo_person['box_person']
        consolidated_players_this_frame.append({
            'player_id': player_id_str,
            'box_display': best_yolo_match_box,
            'center_display': get_center(best_yolo_match_box),
            'type': 'aruco'
        })
        if player_id_str not in tracked_players_data:
            tracked_players_data[player_id_str] = {'first_seen_frame': frame_count, 'performance_log': [], 'color': get_player_color(player_id_str)}
        tracked_players_data[player_id_str]['last_seen_frame'] = frame_count
        tracked_players_data[player_id_str]['current_box'] = best_yolo_match_box

    aruco_identified_yolo_indices = set()
    for aruco_id, aruco_data in detected_aruco_players.items(): # Iterate through ArUco detections
        for idx, yolo_person in enumerate(current_frame_persons_yolo): # Iterate through YOLO detections
             if is_inside_box(aruco_data['center_aruco'], yolo_person['box_person']):
                 # This YOLO person is likely the one wearing the ArUco marker
                 aruco_identified_yolo_indices.add(idx)
                 break # Assume one ArUco marker per person for simplicity

    for idx, yolo_person in enumerate(current_frame_persons_yolo):
        if idx not in aruco_identified_yolo_indices and yolo_person['yolo_tracker_id'] is not None:
            player_id_str = f"Y-{yolo_person['yolo_tracker_id']}"
            consolidated_players_this_frame.append({
                'player_id': player_id_str,
                'box_display': yolo_person['box_person'],
                'center_display': yolo_person['center_person'],
                'type': 'yolo'
            })
            if player_id_str not in tracked_players_data:
                tracked_players_data[player_id_str] = {'first_seen_frame': frame_count, 'performance_log': [], 'color': get_player_color(player_id_str)}
            tracked_players_data[player_id_str]['last_seen_frame'] = frame_count
            tracked_players_data[player_id_str]['current_box'] = yolo_person['box_person']

    for p_info in consolidated_players_this_frame:
        p_id = p_info['player_id']
        p_box = p_info['box_display']
        if p_id in tracked_players_data: 
            p_color = tracked_players_data[p_id]['color']
            cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), p_color, 2)
            cv2.putText(frame, f"ID: {p_id}", (p_box[0], p_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

    # --- Ball Detection ---
    ball_info = None
    ball_detections = object_model(frame, classes=[32], verbose=False)[0] 
    if ball_detections.boxes is not None:
        for det in ball_detections.boxes:
            x1b,y1b,x2b,y2b = map(int, det.xyxy[0]); center_b = get_center((x1b,y1b,x2b,y2b)); conf_b = det.conf.item()
            if is_inside_polygon(center_b, PLAYING_FIELD_POLYGON) and conf_b > 0.3:
                if ball_info is None or conf_b > ball_info['conf']: ball_info = {'box':(x1b,y1b,x2b,y2b),'center':center_b,'conf':conf_b}
                cv2.rectangle(frame, (x1b,y1b), (x2b,y2b), (0,165,255), 2)
                cv2.putText(frame, f"ball {conf_b:.2f}", (x1b,y1b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

    # --- Pose Estimation ---
    pose_results_list = pose_model(frame, verbose=False)[0]
    all_poses_in_frame = []
    if pose_results_list.keypoints is not None and \
       hasattr(pose_results_list.keypoints, 'xy') and pose_results_list.keypoints.xy is not None and \
       hasattr(pose_results_list.keypoints, 'conf') and pose_results_list.keypoints.conf is not None:
        num_poses = len(pose_results_list.keypoints.xy)
        if num_poses > 0 and len(pose_results_list.keypoints.conf) == num_poses : # Ensure conf also has same number of entries
            for i in range(num_poses):
                kpts_tensor = pose_results_list.keypoints.xy[i]
                conf_tensor = pose_results_list.keypoints.conf[i]
                keypoints_dict = get_keypoints_dict(kpts_tensor, conf_tensor)
                if keypoints_dict:
                    core_kpt_name = "nose" if "nose" in keypoints_dict else ("left_hip" if "left_hip" in keypoints_dict else None)
                    if core_kpt_name and keypoints_dict[core_kpt_name].get('xy') is not None and \
                       is_inside_polygon(keypoints_dict[core_kpt_name]['xy'], PLAYING_FIELD_POLYGON):
                        all_poses_in_frame.append(keypoints_dict)

    # --- Associate Poses with Consolidated Players ---
    players_with_pose_and_consolidated_id = []
    for p_info in consolidated_players_this_frame:
        player_id_str = p_info['player_id']
        player_box = p_info['box_display']
        player_center = p_info['center_display']
        best_match_pose_kpts = None
        min_dist_pose = float('inf')
        for pose_kpts in all_poses_in_frame:
            ref_kpt_name = "nose" if "nose" in pose_kpts else ("left_hip" if "left_hip" in pose_kpts else None)
            if ref_kpt_name and pose_kpts[ref_kpt_name].get('xy') is not None:
                kpt_xy = pose_kpts[ref_kpt_name]['xy']
                if is_inside_box(kpt_xy, player_box):
                    dist = get_distance_sq(kpt_xy, player_center)
                    if dist < min_dist_pose:
                        min_dist_pose = dist
                        best_match_pose_kpts = pose_kpts
        if best_match_pose_kpts:
            if player_id_str in tracked_players_data: 
                players_with_pose_and_consolidated_id.append({
                    'player_id': player_id_str,
                    'box': player_box,
                    'keypoints': best_match_pose_kpts,
                    'color': tracked_players_data[player_id_str]['color']
                })
                tracked_players_data[player_id_str]['current_pose'] = best_match_pose_kpts

    if debug_print:
        print(f"  Consolidated Players: {len(consolidated_players_this_frame)}")
        print(f"  Poses in Frame: {len(all_poses_in_frame)}")
        print(f"  Players with Pose & ID: {len(players_with_pose_and_consolidated_id)}")

    # --- Draw Player Body Parts and Skeletons ---
    for player_data in players_with_pose_and_consolidated_id:
        kpts = player_data['keypoints']
        player_color = player_data['color']
        for part_name, kp_names_in_part in BODY_PART_MAP.items():
            avg_pos_coords = []
            for kp_name in kp_names_in_part:
                if kp_name in kpts and kpts[kp_name].get('xy') is not None:
                    pt = kpts[kp_name]['xy']
                    if part_name in ["left_foot", "right_foot", "head"]: avg_pos_coords.append(pt)
            if avg_pos_coords:
                center_of_part = (int(np.mean([p[0] for p in avg_pos_coords])), int(np.mean([p[1] for p in avg_pos_coords])))
                if part_name in ["left_foot", "right_foot"]:
                     cv2.putText(frame, part_name, (center_of_part[0],center_of_part[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, player_color, 1)
                     cv2.circle(frame, center_of_part, 5, player_color, -1)
                elif part_name == "head" and "nose" in kpts and kpts["nose"].get('xy') is not None:
                    cv2.putText(frame, "Head", (kpts["nose"]['xy'][0],kpts["nose"]['xy'][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, player_color, 1)
        for i_skel, j_skel in SKELETON_CONNECTIONS:
            kp_name1,kp_name2 = KEYPOINT_NAMES[i_skel],KEYPOINT_NAMES[j_skel]
            if kp_name1 in kpts and kp_name2 in kpts and \
               kpts[kp_name1].get('xy') is not None and kpts[kp_name2].get('xy') is not None:
                pt1=kpts[kp_name1]['xy']; pt2=kpts[kp_name2]['xy']
                cv2.line(frame, pt1, pt2, player_color, 1)

    # --- Possession Detection ---
    possessor_final_id = None
    min_ball_dist_sq = float('inf')
    POSSESSION_THRESHOLD_SQ = 4500
    if ball_info and ball_info.get('center') is not None:
        ball_center = ball_info['center']
        for player_data in players_with_pose_and_consolidated_id:
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
                    possessor_final_id = player_data['player_id']

    if possessor_final_id is not None and possessor_final_id in tracked_players_data:
        if 'possession_frames' not in tracked_players_data[possessor_final_id]:
            tracked_players_data[possessor_final_id]['possession_frames'] = 0
        tracked_players_data[possessor_final_id]['possession_frames'] += 1
        if 'current_box' in tracked_players_data[possessor_final_id]:
             px1,py1,px2,py2 = tracked_players_data[possessor_final_id]['current_box']
             cv2.rectangle(frame, (px1,py1), (px2,py2), (0,0,255), 3)
             cv2.putText(frame, f"ID {possessor_final_id} In Possession", (px1,py1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # --- Performance Logging Placeholder ---
    for p_data in players_with_pose_and_consolidated_id:
        p_id = p_data['player_id']
        if p_id in tracked_players_data:
            pass # Actual performance logging would happen here

    # --- Show Output ---
    cv2.imshow("YOLO Sports Analysis with ArUco & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- After the loop, print summary ---
print("\n--- Tracked Player Summary (ArUco/YOLO) ---")
for p_id, data in tracked_players_data.items():
    print(f"Player ID (Consolidated): {p_id}")
    print(f"  First seen: frame {data.get('first_seen_frame', 'N/A')}")
    print(f"  Last seen: frame {data.get('last_seen_frame', 'N/A')}")
    if 'possession_frames' in data:
        print(f"  Frames in possession: {data.get('possession_frames', 0)}")
    print(f"  Performance log entries: {len(data.get('performance_log', []))}")