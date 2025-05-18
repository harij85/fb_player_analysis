# g2_api_aws.py (Refactored from g2_mvp_aws.py for FastAPI)
import cv2
import numpy as np
from ultralytics import YOLO
import cv2.aruco as aruco
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # Good practice, though not strictly used for return in this direct refactor
from typing import List, Dict, Tuple, Optional, Any
import base64
import json
import time # For unique request IDs if needed
import os

print("Script g2_api_aws.py starting...")

# --- COCO Keypoints (Unchanged) ---
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
    # ... (rest of your SKELETON_CONNECTIONS) ...
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

# --- Global Models and State (to be initialized in startup) ---
object_model: Optional[YOLO] = None
pose_model: Optional[YOLO] = None
aruco_dict: Optional[aruco.Dictionary] = None
aruco_parameters: Optional[aruco.DetectorParameters] = None

# Global storage for tracked player data (persists for the lifetime of the server process)
# For multi-worker setups, this state won't be shared without a database/cache.
tracked_players_data: Dict[str, Dict[str, Any]] = {}
player_colors: Dict[str, Tuple[int,int,int]] = {}

# --- Helper Functions (Unchanged from your g2_mvp_aws.py) ---
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_aruco_center(corners):
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

def is_inside_polygon(point, polygon): # Polygon is an np.array
    if point is None: return False
    if polygon is None: return True # If no polygon, assume inside
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def get_keypoints_dict(keypoints_tensor, conf_tensor):
    kpts = {}
    if keypoints_tensor is None or conf_tensor is None: return kpts
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(keypoints_tensor) and i < len(conf_tensor):
            if len(keypoints_tensor[i]) < 2: continue # Ensure kpt has x,y
            x_val = keypoints_tensor[i][0]; y_val = keypoints_tensor[i][1]; conf_val = conf_tensor[i]
            x_int = int(x_val.item() if hasattr(x_val, 'item') else x_val)
            y_int = int(y_val.item() if hasattr(y_val, 'item') else y_val)
            conf_float = float(conf_val.item() if hasattr(conf_val, 'item') else conf_val)
            if conf_float > 0.3: kpts[name] = {'xy': (x_int, y_int), 'conf': conf_float}
    return kpts

def get_player_color_api(player_id): # Renamed to avoid conflict if you import another get_player_color
    global player_colors
    if player_id not in player_colors:
        player_colors[player_id] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
    return player_colors[player_id]

def encode_image_base64(frame: np.ndarray) -> str:
    _, img_encoded = cv2.imencode(".jpg", frame)
    return base64.b64encode(img_encoded.tobytes()).decode("utf-8")


# --- FastAPI App Instance ---
app = FastAPI(title="G2 Sports Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global object_model, pose_model, aruco_dict, aruco_parameters
    print("API Startup: Loading models...")
    try:
        object_model = YOLO("yolov8n.pt")
        pose_model = YOLO("yolov8n-pose.pt")
    except Exception as e:
        print(f"FATAL: Error loading YOLO models during startup: {e}")
        traceback.print_exc()
        # Depending on severity, you might want to raise an error to stop FastAPI startup
        raise RuntimeError(f"Could not load YOLO models: {e}") from e
    print("Models loaded.")

    print("API Startup: Initializing ArUco detector...")
    ARUCO_DICT_NAME = "DICT_4X4_100"
    try:
        aruco_dictionary_id_cv_attr = getattr(cv2.aruco, ARUCO_DICT_NAME)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dictionary_id_cv_attr)
    except AttributeError:
        print(f"AttributeError with direct ArUco access for {ARUCO_DICT_NAME}, trying Dictionary_get...")
        try:
            aruco_id_val = cv2.aruco.DICT_4X4_100 # Fallback to a known good one
            if ARUCO_DICT_NAME == "DICT_6X6_250": aruco_id_val = cv2.aruco.DICT_6X6_250 # Example
            aruco_dict = cv2.aruco.Dictionary_get(aruco_id_val)
        except Exception as e_aruco_dict:
            print(f"FATAL: Could not initialize ArUco dictionary {ARUCO_DICT_NAME}: {e_aruco_dict}")
            traceback.print_exc(); raise RuntimeError(f"Could not init ArUco dict: {e_aruco_dict}") from e_aruco_dict
    if aruco_dict is None: raise RuntimeError("FATAL: Failed to load any ArUco dictionary.")
    try: aruco_parameters = cv2.aruco.DetectorParameters_create()
    except AttributeError: aruco_parameters = aruco.DetectorParameters()
    print("ArUco detector initialized.")
    print("--- API Startup Complete ---")


# --- Core Processing Logic (moved from main loop) ---
def analyze_single_frame(
    frame: np.ndarray,
    current_frame_number: int, # Passed by client or generated
    field_polygon_coords: Optional[List[Tuple[int,int]]] = None,
    debug_print_interval: int = 30
    ):
    global tracked_players_data # To update persistent tracking info

    if not all([object_model, pose_model, aruco_dict, aruco_parameters]):
        print("CRITICAL API ERROR: Core models not initialized!")
        # This should ideally not happen if startup was successful
        return {"error": "Server models not ready"}, frame # Return original frame on severe error

    debug_print = (current_frame_number % debug_print_interval == 0)
    if debug_print: print(f"\n--- API Processing frame {current_frame_number} ---")

    frame_height, frame_width = frame.shape[:2]
    
    # Define PLAYING_FIELD_POLYGON for this request
    # This resolves the NameError from your g2_mvp_aws.py
    if field_polygon_coords:
        LOCAL_PLAYING_FIELD_POLYGON = np.array(field_polygon_coords, np.int32)
    else:
        if debug_print: print("  No field polygon provided for request, using full frame.")
        LOCAL_PLAYING_FIELD_POLYGON = np.array(
            [[0,0],[frame_width-1,0],[frame_width-1,frame_height-1],[0,frame_height-1]], 
            np.int32
        )
    
    annotated_frame = frame.copy() # Annotate on a copy
    if LOCAL_PLAYING_FIELD_POLYGON is not None:
        cv2.polylines(annotated_frame, [LOCAL_PLAYING_FIELD_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ArUco Detection
    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    detected_aruco_players = {}
    if ids is not None:
        aruco.drawDetectedMarkers(annotated_frame, corners, ids)
        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0]); marker_corners = corners[i]
            x_coords = marker_corners[0,:,0]; y_coords = marker_corners[0,:,1]
            aruco_box = (int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))
            if not is_inside_polygon(get_center(aruco_box), LOCAL_PLAYING_FIELD_POLYGON): continue
            detected_aruco_players[marker_id] = {'box_aruco': aruco_box, 'center_aruco': get_aruco_center(marker_corners)}

    # YOLO Person Tracking
    yolo_person_detections_with_ids = []
    try:
        yolo_person_results = object_model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)[0]
        raw_tracker_ids = None
        if hasattr(yolo_person_results, 'boxes') and yolo_person_results.boxes is not None and hasattr(yolo_person_results.boxes, 'id') and yolo_person_results.boxes.id is not None:
            raw_tracker_ids = yolo_person_results.boxes.id.int().cpu().tolist()
        if yolo_person_results.boxes is not None:
            for i in range(len(yolo_person_results.boxes)):
                box_data = yolo_person_results.boxes[i]
                x1,y1,x2,y2 = map(int,box_data.xyxy[0]); center = get_center((x1,y1,x2,y2))
                if not is_inside_polygon(center, LOCAL_PLAYING_FIELD_POLYGON): continue
                yolo_tracker_id = raw_tracker_ids[i] if raw_tracker_ids and i < len(raw_tracker_ids) else None
                conf = float(box_data.conf[0].item()) if hasattr(box_data, 'conf') and box_data.conf is not None else 0.0
                yolo_person_detections_with_ids.append({'box_person':(x1,y1,x2,y2), 'center_person':center,'yolo_tracker_id':yolo_tracker_id,'conf':conf})
    except Exception as e_track:
        if debug_print: print(f"  WARNING: YOLO person tracking FAILED in API: {e_track}"); traceback.print_exc()

    # Player Consolidation
    consolidated_players_this_frame_list = [] # List of dicts for players this frame
    for aruco_id, aruco_data in detected_aruco_players.items():
        player_id_str=f"A-{aruco_id}"; best_yolo_match_box=aruco_data['box_aruco']
        min_dist_yolo = float('inf')
        for yolo_person in yolo_person_detections_with_ids:
            if is_inside_box(aruco_data['center_aruco'], yolo_person['box_person']):
                dist = get_distance_sq(aruco_data['center_aruco'], yolo_person['center_person'])
                if dist < min_dist_yolo: min_dist_yolo = dist; best_yolo_match_box = yolo_person['box_person']
        player_info = {'player_id':player_id_str,'box_display':best_yolo_match_box,'center_display':get_center(best_yolo_match_box),'type':'aruco'}
        consolidated_players_this_frame_list.append(player_info)
        if player_id_str not in tracked_players_data:
            tracked_players_data[player_id_str] = {'first_seen_frame': current_frame_number, 'color': get_player_color_api(player_id_str)}
        tracked_players_data[player_id_str]['last_seen_frame'] = current_frame_number
        tracked_players_data[player_id_str]['current_box'] = best_yolo_match_box

    # ... (YOLO-only consolidation logic, adapted)
    for yolo_person in yolo_person_detections_with_ids:
        if yolo_person['yolo_tracker_id'] is not None:
            is_already_aruco = any(p_info['type'] == 'aruco' and calculate_iou(yolo_person['box_person'], p_info['box_display']) > 0.5 for p_info in consolidated_players_this_frame_list)
            if not is_already_aruco:
                player_id_str = f"Y-{yolo_person['yolo_tracker_id']}"
                if not any(p['player_id'] == player_id_str for p in consolidated_players_this_frame_list): # Avoid double add if somehow possible
                    player_info = {'player_id': player_id_str, 'box_display': yolo_person['box_person'],
                                   'center_display': yolo_person['center_person'], 'type': 'yolo'}
                    consolidated_players_this_frame_list.append(player_info)
                    if player_id_str not in tracked_players_data:
                        tracked_players_data[player_id_str] = {'first_seen_frame': current_frame_number, 'color': get_player_color_api(player_id_str)}
                    tracked_players_data[player_id_str]['last_seen_frame'] = current_frame_number
                    tracked_players_data[player_id_str]['current_box'] = yolo_person['box_person']

    # Draw consolidated players
    for p_info in consolidated_players_this_frame_list:
        p_id = p_info['player_id']; p_box = p_info['box_display']
        if p_id in tracked_players_data:
            p_color = tracked_players_data[p_id]['color']
            cv2.rectangle(annotated_frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), p_color, 2)
            cv2.putText(annotated_frame, f"ID: {p_id}", (p_box[0], p_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

    # Ball Detection
    ball_info_dict = None # For returning
    if object_model: # Ensure model is loaded
        ball_detections_yolo = object_model(frame, classes=[32], verbose=False)[0]
        best_ball_candidate = None
        if ball_detections_yolo.boxes is not None:
            for det in ball_detections_yolo.boxes:
                x1b,y1b,x2b,y2b = map(int, det.xyxy[0]); center_b = get_center((x1b,y1b,x2b,y2b)); conf_b = det.conf.item()
                if is_inside_polygon(center_b, LOCAL_PLAYING_FIELD_POLYGON) and conf_b > 0.3:
                    if best_ball_candidate is None or conf_b > best_ball_candidate['conf']:
                        best_ball_candidate = {'box':(x1b,y1b,x2b,y2b),'center':center_b,'conf':conf_b}
            if best_ball_candidate:
                ball_info_dict = best_ball_candidate
                b_box = ball_info_dict['box']
                cv2.rectangle(annotated_frame, (b_box[0],b_box[1]), (b_box[2],b_box[3]), (0,165,255), 2)
                cv2.putText(annotated_frame, f"ball {ball_info_dict['conf']:.2f}", (b_box[0],b_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

    # Pose Estimation
    all_raw_poses_in_frame = []
    if pose_model: # Ensure model is loaded
        pose_results_list = pose_model(frame, verbose=False)[0]
        if pose_results_list.keypoints is not None and hasattr(pose_results_list.keypoints, 'xy') and hasattr(pose_results_list.keypoints, 'conf'):
            # ... (logic from your g2_mvp to populate all_raw_poses_in_frame)
            num_poses = len(pose_results_list.keypoints.xy)
            if num_poses > 0 and (pose_results_list.keypoints.conf is None or len(pose_results_list.keypoints.conf) == num_poses) :
                for i in range(num_poses):
                    kpts_tensor = pose_results_list.keypoints.xy[i]
                    conf_tensor = pose_results_list.keypoints.conf[i] if pose_results_list.keypoints.conf is not None else [0.0] * len(kpts_tensor) # Handle if conf is None
                    keypoints_dict = get_keypoints_dict(kpts_tensor, conf_tensor)
                    if keypoints_dict:
                        core_kpt_name = "nose" if "nose" in keypoints_dict else ("left_hip" if "left_hip" in keypoints_dict else None)
                        if core_kpt_name and keypoints_dict[core_kpt_name].get('xy') and is_inside_polygon(keypoints_dict[core_kpt_name]['xy'], LOCAL_PLAYING_FIELD_POLYGON):
                            all_raw_poses_in_frame.append(keypoints_dict)

    # Associate Poses and Draw
    players_with_pose_list_for_return = [] # List of dicts
    for p_consolidated_info in consolidated_players_this_frame_list:
        player_id = p_consolidated_info['player_id']
        player_box = p_consolidated_info['box_display']
        player_center = p_consolidated_info['center_display']
        player_type = p_consolidated_info['type']
        player_color = tracked_players_data[player_id]['color'] if player_id in tracked_players_data else (255,255,255)

        best_match_pose_kpts = None; min_dist_pose = float('inf')
        # ... (association logic from your g2_mvp) ...
        for pose_kpts in all_raw_poses_in_frame:
            ref_kpt_name = "nose" if "nose" in pose_kpts else ("left_hip" if "left_hip" in pose_kpts else None)
            if ref_kpt_name and pose_kpts[ref_kpt_name].get('xy'):
                kpt_xy = pose_kpts[ref_kpt_name]['xy']
                if is_inside_box(kpt_xy, player_box):
                    dist = get_distance_sq(kpt_xy, player_center)
                    if dist < min_dist_pose: min_dist_pose = dist; best_match_pose_kpts = pose_kpts
        
        player_pose_data_for_return = {
            "player_id": player_id, "box_display": player_box,
            "center_display": player_center, "type": player_type,
            "keypoints": {} # Empty if no pose
        }

        if best_match_pose_kpts:
            player_pose_data_for_return["keypoints"] = best_match_pose_kpts
            if player_id in tracked_players_data:
                tracked_players_data[player_id]['current_pose'] = best_match_pose_kpts # For global state

            # Draw skeleton if pose found
            for i_skel, j_skel in SKELETON_CONNECTIONS:
                kp_name1,kp_name2 = KEYPOINT_NAMES[i_skel],KEYPOINT_NAMES[j_skel]
                if kp_name1 in best_match_pose_kpts and kp_name2 in best_match_pose_kpts and \
                   best_match_pose_kpts[kp_name1].get('xy') and best_match_pose_kpts[kp_name2].get('xy'):
                    pt1=best_match_pose_kpts[kp_name1]['xy']; pt2=best_match_pose_kpts[kp_name2]['xy']
                    cv2.line(annotated_frame, pt1, pt2, player_color, 1)
            # Draw keypoints
            for kp_name_draw, kp_data_draw in best_match_pose_kpts.items():
                if kp_data_draw.get('xy'):
                    cv2.circle(annotated_frame, kp_data_draw['xy'], 2, player_color, -1)
        
        players_with_pose_list_for_return.append(player_pose_data_for_return)


    # Possession Detection
    possessor_final_id = None
    # ... (possession logic from your g2_mvp, using players_with_pose_list_for_return and ball_info_dict) ...
    min_ball_dist_sq = float('inf'); POSSESSION_THRESHOLD_SQ = 4500
    if ball_info_dict and ball_info_dict.get('center') is not None:
        ball_center = ball_info_dict['center']
        for player_data in players_with_pose_list_for_return: # Iterates list of dicts
            kpts = player_data['keypoints'] # This is the raw keypoints_dict
            if not kpts: continue # Skip if no pose associated
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

    if possessor_final_id and possessor_final_id in tracked_players_data:
        if 'possession_frames' not in tracked_players_data[possessor_final_id]:
            tracked_players_data[possessor_final_id]['possession_frames'] = 0
        tracked_players_data[possessor_final_id]['possession_frames'] += 1
        if 'current_box' in tracked_players_data[possessor_final_id]:
             px1,py1,px2,py2 = tracked_players_data[possessor_final_id]['current_box']
             cv2.rectangle(annotated_frame, (px1,py1), (px2,py2), (0,0,255), 3)
             cv2.putText(annotated_frame, f"ID {possessor_final_id} Has Ball", (px1,py1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Prepare results
    results_dict = {
        "frame_number": current_frame_number,
        "players": players_with_pose_list_for_return, # List of player dicts (with pose if found)
        "ball": ball_info_dict,
        "possessor_id": possessor_final_id,
        "num_consolidated_players": len(consolidated_players_this_frame_list),
        "num_raw_poses_in_frame": len(all_raw_poses_in_frame),
        "num_players_with_pose": sum(1 for p in players_with_pose_list_for_return if p["keypoints"])
    }
    return results_dict, annotated_frame


# --- FastAPI Endpoint ---
@app.post("/process_g2_frame/") # Changed endpoint name to reflect script
async def process_g2_frame_endpoint(
    image_file: UploadFile = File(...),
    frame_number: int = Form(0), # Client should send this
    playing_field_polygon_coords_json: Optional[str] = Form(None),
    return_annotated_image: bool = Form(True)
):
    request_time_start = time.time()
    req_id_suffix = int(request_time_start*1000)%10000
    if frame_number % 30 == 0: print(f"Req-{req_id_suffix}: API Frame {frame_number}")

    contents = await image_file.read()
    frame_cv = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame_cv is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    polygon_coords_list: Optional[List[Tuple[int, int]]] = None
    if playing_field_polygon_coords_json:
        try:
            parsed_json = json.loads(playing_field_polygon_coords_json)
            if not (isinstance(parsed_json, list) and len(parsed_json) >= 3 and
                    all(isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(c, (int, float)) for c in pt) for pt in parsed_json)):
                raise ValueError("Invalid polygon format.")
            polygon_coords_list = [(int(pt[0]), int(pt[1])) for pt in parsed_json]
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid polygon JSON: {e}")

    try:
        analysis_results_dict, annotated_frame = analyze_single_frame(
            frame_cv,
            current_frame_number=frame_number,
            field_polygon_coords=polygon_coords_list
        )
        
        if "error" in analysis_results_dict: # Check if processing function returned an error
             raise HTTPException(status_code=500, detail=analysis_results_dict["error"])

        annotated_image_b64 = None
        if return_annotated_image and annotated_frame is not None:
            annotated_image_b64 = encode_image_base64(annotated_frame)
        
        analysis_results_dict["annotated_image_base64"] = annotated_image_b64
        
        if frame_number % 30 == 0:
            print(f"Req-{req_id_suffix}: API Frame {frame_number} processed in {time.time() - request_time_start:.4f} secs.")
        return analysis_results_dict

    except HTTPException as e: raise e
    except Exception as e:
        print(f"Req-{req_id_suffix}: !!! UNEXPECTED error processing API frame {frame_number}: {e} !!!"); traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn server for g2_api_aws.py ...")
    # Ensure bytetrack.yaml is in the working directory or accessible by YOLO.
    uvicorn.run("g2_api_aws:app", host="0.0.0.0", port=8000, workers=1)