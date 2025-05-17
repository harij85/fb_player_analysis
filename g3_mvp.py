import cv2
import numpy as np
from ultralytics import YOLO
import cv2.aruco as aruco
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Any
import base64
import io
import time
from collections import deque # Added for BallTracker/Annotator
import supervision as sv # Added for BallTracker/Annotator
import json # For parsing JSON from form if needed

from fastapi.middleware.cors import CORSMiddleware

from feedback import Get_Perf_From_Pose





# --- Configuration & Constants (from your script) ---
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

# --- Ball Tracking and Annotation Classes (using supervision) ---
class BallAnnotator:
    def __init__(self, radius: int, buffer_size: int = 10, thickness: int = 2): # Increased default buffer for longer trail
        self.color_palette = sv.ColorPalette.from_matplotlib('cool', buffer_size) # Changed palette for variety
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i <= 1: # Handle single point case
            return self.radius
        # Linear interpolation from radius/N to radius
        min_radius_factor = 0.2 # Smallest circle will be 20% of max radius
        return int(self.radius * (min_radius_factor + (1-min_radius_factor) * (i / (max_i -1)) ) )


    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0: # If no ball detected, nothing to add to buffer or draw
            # Optionally, draw existing trail even if current detection is lost
            # For now, only add to buffer if there is a new detection
            pass
        else:
            # Get bottom center of the *first (and presumably only)* detected ball
            xy_current_ball = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0].astype(int)
            self.buffer.append(tuple(xy_current_ball)) # Ensure it's a tuple for storage

        annotated_frame = frame.copy() # Work on a copy to avoid modifying original if passed around
        for i, center_tuple in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            annotated_frame = cv2.circle(
                img=annotated_frame,
                center=center_tuple, # Already a tuple
                radius=interpolated_radius,
                color=color.as_bgr(),
                thickness=self.thickness
            )
        return annotated_frame


class BallTracker:
    def __init__(self, buffer_size: int = 10): # Buffer of recent ball positions
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            # If no current detections, we might still want to manage the buffer
            # (e.g., if it relies on time, but here it's just based on new detections)
            # For now, just return empty if no candidates.
            # Optionally, could try to predict based on buffer, but that's more complex.
            return sv.Detections.empty()

        # Get center coordinates of all current candidate ball detections
        current_centers_xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        
        # Add current detections' centers to the buffer if there are any
        # The buffer stores lists of centers from each frame it saw detections
        if len(current_centers_xy) > 0:
            self.buffer.append(current_centers_xy)

        if not self.buffer: # Buffer is empty, cannot calculate centroid
            if len(detections) > 0: # If buffer was empty but now we have detections
                 # Heuristic: pick the first one, or highest confidence if available & multiple
                 # For simplicity, if buffer is empty, we pick based on current frame only
                 # Or if only one detection, that's our best bet
                if len(detections) == 1:
                    return detections
                else: # Multiple detections, buffer empty, pick highest confidence
                    if detections.confidence is not None:
                        best_idx = np.argmax(detections.confidence)
                        return detections[[best_idx]]
                    else: # No confidence, pick first
                        return detections[[0]] 
            return sv.Detections.empty() # No detections, buffer empty

        # Calculate centroid of all points in the buffer (recent history)
        # Concatenate all arrays of centers in the buffer into one large array
        all_historical_centers = np.concatenate(list(self.buffer), axis=0)
        centroid = np.mean(all_historical_centers, axis=0)

        # Calculate distances from each current detection's center to this historical centroid
        distances = np.linalg.norm(current_centers_xy - centroid, axis=1)
        
        # Select the current detection that is closest to the historical centroid
        index_of_closest = np.argmin(distances)
        return detections[[index_of_closest]]


# --- Global storage for persistent data across API calls ---
player_colors = {}

# --- Helper Functions (from your script) ---
def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_aruco_center(corners) -> Tuple[int, int]:
    cX = int(np.mean(corners[0,:,0]))
    cY = int(np.mean(corners[0,:,1]))
    return (cX, cY)

def get_distance_sq(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> float:
    if p1 is None or p2 is None: return float('inf')
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def is_inside_box(point: Optional[Tuple[int, int]], box: Optional[Tuple[int, int, int, int]]) -> bool:
    if point is None or box is None: return False
    x, y = point; x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_inside_polygon(point: Optional[Tuple[int, int]], polygon: Optional[np.ndarray]) -> bool:
    if point is None: return False
    if polygon is None: return True
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

def get_keypoints_dict(keypoints_tensor, conf_tensor) -> Dict[str, Dict[str, Any]]:
    kpts = {}
    if keypoints_tensor is None or conf_tensor is None: return kpts
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(keypoints_tensor) and i < len(conf_tensor):
            x, y = keypoints_tensor[i]; conf = conf_tensor[i]
            if conf > 0.3: kpts[name] = {'xy': (int(x), int(y)), 'conf': float(conf.item())}
    return kpts

def get_player_color(player_id: str) -> Tuple[int, int, int]:
    if player_id not in player_colors:
        player_colors[player_id] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
    return player_colors[player_id]


# --- FastAPI App and Global Model Initialization ---
app = FastAPI(title="Football Performance Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and ArUco setup
object_model: Optional[YOLO] = None
pose_model: Optional[YOLO] = None
aruco_dict: Optional[aruco.Dictionary] = None
aruco_parameters: Optional[aruco.DetectorParameters] = None
ball_tracker: Optional[BallTracker] = None
ball_annotator: Optional[BallAnnotator] = None

@app.on_event("startup")
async def load_models_and_setup():
    global object_model, pose_model, aruco_dict, aruco_parameters, ball_tracker, ball_annotator
    print("Loading YOLO models...")
    try:
        object_model = YOLO("yolov8n.pt")
        pose_model = YOLO("yolov8n-pose.pt")
    except Exception as e:
        print(f"Fatal: Error loading YOLO models: {e}")
        raise RuntimeError(f"Could not load YOLO models: {e}") from e
    print("YOLO Models loaded.")

    print("Initializing ArUco detector...")
    ARUCO_DICT_NAME = "DICT_4X4_100"
    try:
        aruco_dictionary_id_cv_attr = getattr(cv2.aruco, ARUCO_DICT_NAME)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dictionary_id_cv_attr)
    except AttributeError:
        print(f"AttributeError with direct access to {ARUCO_DICT_NAME}, trying Dictionary_get method...")
        try:
            aruco_dict_map = {
                "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                "DICT_6X6_250": cv2.aruco.DICT_6X6_250
            } # Added more common ones
            aruco_id_val = aruco_dict_map.get(ARUCO_DICT_NAME)
            if aruco_id_val is not None:
                 aruco_dict = cv2.aruco.Dictionary_get(aruco_id_val)
            else:
                print(f"ArUco dictionary {ARUCO_DICT_NAME} not in map, using DICT_4X4_100")
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        except Exception as e_aruco_dict:
            print(f"Fatal: Could not initialize ArUco dictionary {ARUCO_DICT_NAME}: {e_aruco_dict}")
            raise RuntimeError(f"Could not initialize ArUco: {e_aruco_dict}") from e_aruco_dict

    if aruco_dict is None:
        print("Fatal: Failed to load any ArUco dictionary.")
        raise RuntimeError("Failed to load ArUco dictionary")

    aruco_parameters = aruco.DetectorParameters()
    print("ArUco detector initialized.")

    print("Initializing Ball Tracker and Annotator...")
    # Configure these parameters as needed
    ball_tracker = BallTracker(buffer_size=15) # Increased buffer for smoother tracking
    ball_annotator = BallAnnotator(radius=10, buffer_size=15, thickness=2) # Trail radius
    print("Ball Tracker and Annotator initialized.")


# --- Pydantic Models for API Input/Output ---
class KeypointData(BaseModel):
    xy: Tuple[int, int]
    conf: float

class PlayerPoseInfo(BaseModel):
    player_id: str
    box_display: Tuple[int, int, int, int]
    center_display: Tuple[int, int]
    keypoints: Dict[str, KeypointData]
    color: Tuple[int, int, int]
    type: str # 'aruco' or 'yolo'

class BallInfo(BaseModel):
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    conf: float

class FrameAnalysisResult(BaseModel):
    players: List[PlayerPoseInfo] = []
    ball: Optional[BallInfo] = None
    possessor_id: Optional[str] = None
    annotated_image_base64: Optional[str] = None
    num_consolidated_players: int = 0
    num_poses_in_frame: int = 0
    num_players_with_pose_and_id: int = 0
    performance_feedback: Optional[List[Dict[str, Any]]] = None


# --- Main Processing Function (adapted from your loop) ---
def process_frame_for_analysis(
    frame: np.ndarray,
    playing_field_polygon_coords: Optional[List[Tuple[int,int]]] = None,
    return_annotated_image: bool = True
) -> FrameAnalysisResult:
    
    if object_model is None or pose_model is None or aruco_dict is None or \
       aruco_parameters is None or ball_tracker is None or ball_annotator is None: # Check new additions
        raise HTTPException(status_code=503, detail="Models or helpers not loaded/initialized.")

    current_frame_player_tracking_info = {}
    frame_height, frame_width = frame.shape[:2]
    
    PLAYING_FIELD_POLYGON: Optional[np.ndarray] = None
    if playing_field_polygon_coords:
        PLAYING_FIELD_POLYGON = np.array(playing_field_polygon_coords, np.int32)
    else:
        PLAYING_FIELD_POLYGON = np.array(
            [[0,0],[frame_width-1,0],[frame_width-1,frame_height-1],[0,frame_height-1]], np.int32)

    debug_print = True 
    if debug_print: print(f"\n--- Processing frame ---")

    if PLAYING_FIELD_POLYGON is not None and return_annotated_image:
        cv2.polylines(frame, [PLAYING_FIELD_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- ArUco Marker Detection --- (No changes here)
    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    detected_aruco_players = {}
    if ids is not None:
        if return_annotated_image: aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])
            marker_corners = corners[i]
            x_coords = marker_corners[0,:,0]; y_coords = marker_corners[0,:,1]
            aruco_box = (int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))
            aruco_center = get_aruco_center(marker_corners)
            detected_aruco_players[marker_id] = {
                'box_aruco': aruco_box, 'center_aruco': aruco_center, 'corners': marker_corners
            }

    # --- YOLO Object Detection and Tracking (for Persons) --- (No changes here)
    try:
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
            yolo_tracker_id = None
            if yolo_ids_present and yolo_tracked_results.boxes.id is not None and i < len(yolo_tracked_results.boxes.id):
                 yolo_tracker_id_tensor = yolo_tracked_results.boxes.id[i]
                 if yolo_tracker_id_tensor is not None: yolo_tracker_id = int(yolo_tracker_id_tensor.item())

            if not is_inside_polygon(center, PLAYING_FIELD_POLYGON): continue
            current_frame_persons_yolo.append({
                'box_person': (x1, y1, x2, y2), 'center_person': center,
                'yolo_tracker_id': yolo_tracker_id, 'conf': box_data.conf.item()
            })

    # --- Consolidate Player Information --- (No changes here)
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
        player_info = {'player_id': player_id_str, 'box_display': best_yolo_match_box,
                       'center_display': get_center(best_yolo_match_box), 'type': 'aruco',
                       'color': get_player_color(player_id_str)}
        consolidated_players_this_frame.append(player_info)
        current_frame_player_tracking_info[player_id_str] = {'current_box': best_yolo_match_box, 'color': player_info['color']}

    aruco_identified_yolo_indices = set()
    for aruco_data in detected_aruco_players.values():
        for idx, yolo_person in enumerate(current_frame_persons_yolo):
             if is_inside_box(aruco_data['center_aruco'], yolo_person['box_person']):
                 aruco_identified_yolo_indices.add(idx); break
    for idx, yolo_person in enumerate(current_frame_persons_yolo):
        if idx not in aruco_identified_yolo_indices and yolo_person['yolo_tracker_id'] is not None:
            player_id_str = f"Y-{yolo_person['yolo_tracker_id']}"
            player_info = {'player_id': player_id_str, 'box_display': yolo_person['box_person'],
                           'center_display': yolo_person['center_person'], 'type': 'yolo',
                           'color': get_player_color(player_id_str)}
            consolidated_players_this_frame.append(player_info)
            current_frame_player_tracking_info[player_id_str] = {'current_box': yolo_person['box_person'], 'color': player_info['color']}

    if return_annotated_image:
        for p_info in consolidated_players_this_frame:
            p_color = p_info['color']; p_box = p_info['box_display']
            cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), p_color, 2)
            cv2.putText(frame, f"ID: {p_info['player_id']}", (p_box[0], p_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

    # --- Ball Detection & Tracking (REFACTORED) ---
    ball_data_dict: Optional[Dict[str, Any]] = None
    ball_result: Optional[BallInfo] = None 

    yolo_ball_results = object_model(frame, classes=[32], verbose=False)[0] # class 32 is 'sports ball'

    raw_ball_xyxy = []
    raw_ball_conf = []
    raw_ball_class_id = [] # Though tracker doesn't use it, good practice

    if yolo_ball_results.boxes is not None:
        for box in yolo_ball_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = get_center((x1, y1, x2, y2))
            conf = float(box.conf[0].item()) # box.conf is a tensor
            cls_id = int(box.cls[0].item())   # box.cls is a tensor

            # Filter by confidence and playing field *before* giving to tracker
            if conf > 0.25 and is_inside_polygon(center, PLAYING_FIELD_POLYGON): # Lowered conf slightly for more candidates
                raw_ball_xyxy.append([x1, y1, x2, y2])
                raw_ball_conf.append(conf)
                raw_ball_class_id.append(cls_id)

    candidate_sv_detections = sv.Detections.empty()
    if raw_ball_xyxy: 
        candidate_sv_detections = sv.Detections(
            xyxy=np.array(raw_ball_xyxy),
            confidence=np.array(raw_ball_conf),
            class_id=np.array(raw_ball_class_id)
        )
    
    # Ensure ball_tracker is used (it should be initialized by FastAPI startup)
    tracked_ball_sv_detections = ball_tracker.update(candidate_sv_detections)

    if len(tracked_ball_sv_detections) > 0:
        # BallTracker returns the single best match
        tracked_ball_data = tracked_ball_sv_detections[0] # sv.Detections object for the single ball
        
        b_x1, b_y1, b_x2, b_y2 = map(int, tracked_ball_data.xyxy[0])
        # Confidence might be from the original detection or tracker might not preserve it explicitly this way
        b_conf = float(tracked_ball_data.confidence[0]) if tracked_ball_data.confidence is not None and len(tracked_ball_data.confidence)>0 else 0.5 
        b_center = get_center((b_x1, b_y1, b_x2, b_y2))

        ball_data_dict = {'box': (b_x1, b_y1, b_x2, b_y2), 'center': b_center, 'conf': b_conf}
        ball_result = BallInfo(**ball_data_dict)

        if return_annotated_image and ball_annotator:
            # Pass the sv.Detections object FOR THE SINGLE TRACKED BALL to the annotator
            frame = ball_annotator.annotate(frame, tracked_ball_sv_detections) 
            # The old rectangle for the ball is now replaced by the trail from ball_annotator

    # --- Pose Estimation --- (No changes here)
    pose_results_list = pose_model(frame, verbose=False)[0]
    all_poses_in_frame = [] 
    if pose_results_list.keypoints is not None and \
       hasattr(pose_results_list.keypoints, 'xy') and pose_results_list.keypoints.xy is not None and \
       hasattr(pose_results_list.keypoints, 'conf') and pose_results_list.keypoints.conf is not None:
        num_poses = len(pose_results_list.keypoints.xy)
        if num_poses > 0 and len(pose_results_list.keypoints.conf) == num_poses :
            for i in range(num_poses):
                keypoints_dict = get_keypoints_dict(pose_results_list.keypoints.xy[i], pose_results_list.keypoints.conf[i])
                if keypoints_dict:
                    core_kpt_name = "nose" if "nose" in keypoints_dict else ("left_hip" if "left_hip" in keypoints_dict else None)
                    if core_kpt_name and keypoints_dict[core_kpt_name].get('xy') is not None and \
                       is_inside_polygon(keypoints_dict[core_kpt_name]['xy'], PLAYING_FIELD_POLYGON):
                        all_poses_in_frame.append(keypoints_dict)

    # --- Associate Poses with Consolidated Players --- (No changes here)
    players_with_pose_result: List[PlayerPoseInfo] = [] 
    for p_info_consolidated in consolidated_players_this_frame:
        player_id_str, player_box, player_center, player_type, player_color = \
            p_info_consolidated['player_id'], p_info_consolidated['box_display'], \
            p_info_consolidated['center_display'], p_info_consolidated['type'], p_info_consolidated['color']
        best_match_pose_kpts = None; min_dist_pose = float('inf')
        for pose_kpts_dict in all_poses_in_frame:
            ref_kpt_name = "nose" if "nose" in pose_kpts_dict else ("left_hip" if "left_hip" in pose_kpts_dict else None)
            if ref_kpt_name and pose_kpts_dict[ref_kpt_name].get('xy') is not None:
                kpt_xy = pose_kpts_dict[ref_kpt_name]['xy']
                if is_inside_box(kpt_xy, player_box):
                    dist = get_distance_sq(kpt_xy, player_center)
                    if dist < min_dist_pose: min_dist_pose = dist; best_match_pose_kpts = pose_kpts_dict
        if best_match_pose_kpts:
            pydantic_kpts = {name: KeypointData(**data) for name, data in best_match_pose_kpts.items()}
            player_pose_data_item = PlayerPoseInfo(player_id=player_id_str, box_display=player_box,
                center_display=player_center, keypoints=pydantic_kpts, color=player_color, type=player_type)
            players_with_pose_result.append(player_pose_data_item)
            if player_id_str in current_frame_player_tracking_info:
                 current_frame_player_tracking_info[player_id_str]['current_pose'] = best_match_pose_kpts

    if debug_print:
        print(f"  Consolidated Players: {len(consolidated_players_this_frame)}")
        print(f"  Poses in Frame: {len(all_poses_in_frame)}")
        print(f"  Players with Pose & ID: {len(players_with_pose_result)}")

    # --- Draw Player Body Parts and Skeletons --- (No changes here)
    if return_annotated_image:
        for player_data_pydantic in players_with_pose_result:
            original_kpts_dict = None
            if player_data_pydantic.player_id in current_frame_player_tracking_info and \
               'current_pose' in current_frame_player_tracking_info[player_data_pydantic.player_id]:
                original_kpts_dict = current_frame_player_tracking_info[player_data_pydantic.player_id]['current_pose']
            if original_kpts_dict:
                p_color = player_data_pydantic.color
                for part_name, kp_names_in_part in BODY_PART_MAP.items():
                    avg_pos_coords = []
                    for kp_name in kp_names_in_part:
                        if kp_name in original_kpts_dict and original_kpts_dict[kp_name].get('xy') is not None:
                            pt = original_kpts_dict[kp_name]['xy']
                            if part_name in ["left_foot", "right_foot", "head"]: avg_pos_coords.append(pt)
                    if avg_pos_coords:
                        center_of_part = (int(np.mean([p[0] for p in avg_pos_coords])), int(np.mean([p[1] for p in avg_pos_coords])))
                        if part_name in ["left_foot", "right_foot"]:
                             cv2.putText(frame, part_name, (center_of_part[0],center_of_part[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, p_color, 1)
                             cv2.circle(frame, center_of_part, 5, p_color, -1)
                        elif part_name == "head" and "nose" in original_kpts_dict and original_kpts_dict["nose"].get('xy') is not None:
                            cv2.putText(frame, "Head", (original_kpts_dict["nose"]['xy'][0],original_kpts_dict["nose"]['xy'][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, p_color, 1)
                for i_skel, j_skel in SKELETON_CONNECTIONS:
                    kp_name1,kp_name2 = KEYPOINT_NAMES[i_skel],KEYPOINT_NAMES[j_skel]
                    if kp_name1 in original_kpts_dict and kp_name2 in original_kpts_dict and \
                       original_kpts_dict[kp_name1].get('xy') is not None and original_kpts_dict[kp_name2].get('xy') is not None:
                        pt1=original_kpts_dict[kp_name1]['xy']; pt2=original_kpts_dict[kp_name2]['xy']
                        cv2.line(frame, pt1, pt2, p_color, 1)

    # --- Possession Detection --- (Ball data now comes from tracked ball)
    possessor_final_id: Optional[str] = None
    min_ball_dist_sq = float('inf')
    POSSESSION_THRESHOLD_SQ = 4500 
    if ball_result and ball_result.center: # ball_result is from the new tracker
        ball_center = ball_result.center
        for player_data_pydantic in players_with_pose_result: 
            original_kpts_dict = None 
            if player_data_pydantic.player_id in current_frame_player_tracking_info and \
               'current_pose' in current_frame_player_tracking_info[player_data_pydantic.player_id]:
                original_kpts_dict = current_frame_player_tracking_info[player_data_pydantic.player_id]['current_pose']
            if original_kpts_dict:
                relevant_kpt_names = ["left_ankle", "right_ankle", "left_knee", "right_knee", "left_wrist", "right_wrist"] # Added wrists for handball
                closest_player_part_dist_sq = float('inf')
                for kp_name in relevant_kpt_names:
                    if kp_name in original_kpts_dict and original_kpts_dict[kp_name].get('xy') is not None:
                        part_xy = original_kpts_dict[kp_name]['xy']
                        dist_sq = get_distance_sq(part_xy, ball_center)
                        if dist_sq < closest_player_part_dist_sq: closest_player_part_dist_sq = dist_sq
                if closest_player_part_dist_sq < POSSESSION_THRESHOLD_SQ:
                    if closest_player_part_dist_sq < min_ball_dist_sq:
                        min_ball_dist_sq = closest_player_part_dist_sq
                        possessor_final_id = player_data_pydantic.player_id

    if return_annotated_image and possessor_final_id is not None and possessor_final_id in current_frame_player_tracking_info:
        player_track_info = current_frame_player_tracking_info[possessor_final_id]
        if 'current_box' in player_track_info:
             px1,py1,px2,py2 = player_track_info['current_box']
             cv2.rectangle(frame, (px1,py1), (px2,py2), (0,0,255), 3) 
             cv2.putText(frame, f"ID {possessor_final_id} In Possession", (px1,py1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # --- Prepare annotated image for response ---
    annotated_image_b64_str: Optional[str] = None
    if return_annotated_image:
        _, img_encoded = cv2.imencode(".jpg", frame)
        annotated_image_b64_str = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

    return FrameAnalysisResult(
        players=players_with_pose_result, ball=ball_result, possessor_id=possessor_final_id,
        annotated_image_base64=annotated_image_b64_str,
        num_consolidated_players=len(consolidated_players_this_frame),
        num_poses_in_frame=len(all_poses_in_frame),
        num_players_with_pose_and_id=len(players_with_pose_result))


# --- FastAPI Endpoint ---
@app.post("/process_frame/", response_model=FrameAnalysisResult)
async def analyze_frame_endpoint(
    image_file: UploadFile = File(..., description="Image file to process (jpg, png, etc.)"),
    playing_field_polygon_coords_json: Optional[str] = Form(None, description="JSON string of polygon coords [[x1,y1],...]. If None, full frame."),
    return_annotated_image: bool = Form(True, description="Whether to return annotated image.")
):
    start_time = time.time()
    contents = await image_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame_cv is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    polygon_coords_list: Optional[List[Tuple[int, int]]] = None
    if playing_field_polygon_coords_json:
        try:
            polygon_coords_list = json.loads(playing_field_polygon_coords_json)

            # ✅ Fixed validation logic for polygon format
            if not isinstance(polygon_coords_list, list) or \
               not all(isinstance(pt, (list, tuple)) and len(pt) == 2 and
                       all(isinstance(coord, (int, float)) for coord in pt) for pt in polygon_coords_list):
                raise ValueError("Invalid polygon format.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid playing_field_polygon_coords_json: {e}")

    try:
        analysis_result = process_frame_for_analysis(
            frame_cv,
            playing_field_polygon_coords=polygon_coords_list,
            return_annotated_image=return_annotated_image
        )

        # 🧠 Generate feedback using helper
        if analysis_result.ball and analysis_result.ball.center:
            feedback_output = Get_Perf_From_Pose(analysis_result.players, analysis_result.ball.center)
            analysis_result.performance_feedback = feedback_output
            print("✓ Generated performance feedback")
        else:
            analysis_result.performance_feedback = []
            print("⚠️ Skipping feedback, no ball center found.")

        end_time = time.time()
        print(f"Frame processed in {end_time - start_time:.4f} seconds.")
        return analysis_result

    except HTTPException as e:
        raise e
    except RuntimeError as e:
        print(f"Runtime error during processing: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    # Ensure you have installed: pip install fastapi "uvicorn[standard]" python-multipart opencv-python ultralytics numpy opencv-contrib-python supervision
    uvicorn.run(app, host="0.0.0.0", port=8000)