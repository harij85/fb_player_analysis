# v5_mvp_pass_detection.py (Integrates Pass Detection LSTM)
import cv2
import numpy as np
from ultralytics import YOLO
import cv2.aruco as aruco
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Any
import base64
import json
import time
import os
from pathlib import Path
from collections import deque, defaultdict # Added defaultdict
import supervision as sv
import traceback
import torch
import torch.nn as nn

# Import from custom modules (assuming they are in the same directory or Python path)
from feedback import Get_Perf_From_Pose
from pose_ball_tracking_kalman import PlayerPoseSmoother, BallMotionAnalyzer, KalmanFilter2D

# --- Constants ---
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# Using full SKELETON_CONNECTIONS for better visualization
SKELETON_CONNECTIONS = [
    (KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("left_shoulder")), (KEYPOINT_NAMES.index("right_hip"), KEYPOINT_NAMES.index("right_shoulder")),
    (KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("right_shoulder")), (KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("right_hip")),
    (KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("left_elbow")), (KEYPOINT_NAMES.index("left_elbow"), KEYPOINT_NAMES.index("left_wrist")),
    (KEYPOINT_NAMES.index("right_shoulder"), KEYPOINT_NAMES.index("right_elbow")), (KEYPOINT_NAMES.index("right_elbow"), KEYPOINT_NAMES.index("right_wrist")),
    (KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("left_knee")), (KEYPOINT_NAMES.index("left_knee"), KEYPOINT_NAMES.index("left_ankle")),
    (KEYPOINT_NAMES.index("right_hip"), KEYPOINT_NAMES.index("right_knee")), (KEYPOINT_NAMES.index("right_knee"), KEYPOINT_NAMES.index("right_ankle")),
    (KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("nose")), (KEYPOINT_NAMES.index("right_shoulder"), KEYPOINT_NAMES.index("nose")),
    (KEYPOINT_NAMES.index("nose"), KEYPOINT_NAMES.index("left_eye")), (KEYPOINT_NAMES.index("nose"), KEYPOINT_NAMES.index("right_eye")),
    (KEYPOINT_NAMES.index("left_eye"), KEYPOINT_NAMES.index("left_ear")), (KEYPOINT_NAMES.index("right_eye"), KEYPOINT_NAMES.index("right_ear")),
]
FRAME_TIME_DELTA = 1.0 / 30.0 # Adjust if your typical video FPS is different

# --- Configuration for Pass Detection Model ---
PASS_MODEL_INPUT_SIZE = 16 # Number of features your LSTM expects (MUST MATCH TRAINING)
PASS_MODEL_HIDDEN_SIZE = 64
PASS_MODEL_NUM_LAYERS = 2
PASS_MODEL_SEQUENCE_LENGTH = 15 # Sequence length model was trained on
PASS_MODEL_PATH = "./pass_data_for_training/pass_detector_lstm_v2.pth" # Path to your trained model
PASS_DETECTION_THRESHOLD = 0.7 # Confidence threshold to declare a pass (tune this)

# --- Ball Tracker & Annotator Classes (Unchanged from previous) ---
class BallAnnotator:
    def __init__(self, radius: int, buffer_size: int = 10, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib('cool', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness
    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i <= 1: return self.radius
        min_radius_factor = 0.2
        return int(self.radius * (min_radius_factor + (1 - min_radius_factor) * (i / (max_i - 1))))
    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) > 0:
            xy_current_ball = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0].astype(int)
            self.buffer.append(tuple(xy_current_ball))
        annotated_frame = frame.copy()
        for i, center_tuple in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            annotated_frame = cv2.circle(img=annotated_frame, center=center_tuple, radius=interpolated_radius, color=color.as_bgr(), thickness=self.thickness)
        return annotated_frame

class BallTracker:
    def __init__(self, buffer_size: int = 10): self.buffer = deque(maxlen=buffer_size)
    def update(self, detections: sv.Detections) -> sv.Detections: # Unchanged
        if len(detections) == 0: return sv.Detections.empty()
        current_centers_xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        if len(current_centers_xy) > 0: self.buffer.append(current_centers_xy)
        if not self.buffer:
            if len(detections) > 0:
                if len(detections) == 1: return detections
                if detections.confidence is not None and len(detections.confidence) > 0: return detections[[np.argmax(detections.confidence)]]
                return detections[[0]]
            return sv.Detections.empty()
        centroid = np.mean(np.concatenate(list(self.buffer), axis=0), axis=0)
        distances = np.linalg.norm(current_centers_xy - centroid, axis=1)
        return detections[[np.argmin(distances)]]

# --- Global storage ---
player_colors = {}
# Buffer for player features for pass detection: {player_id: deque([(frame_num, features_list), ...])}
# We'll also need previous frame data for velocity calculation for EACH player
player_feature_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=PASS_MODEL_SEQUENCE_LENGTH))
player_previous_frame_data: Dict[str, Dict[str, Any]] = {} # {player_id: {'player_info': {...}, 'ball_info': {...}}}

# --- Helper Functions (including feature extraction) ---
def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]: # Unchanged
    x1, y1, x2, y2 = bbox; return (int((x1 + x2) / 2), int((y1 + y2) / 2))
def get_aruco_center(corners) -> Tuple[int, int]: # Unchanged
    return (int(np.mean(corners[0,:,0])), int(np.mean(corners[0,:,1])))
def get_distance_sq(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> float: # Unchanged
    if p1 is None or p2 is None: return float('inf')
    if not (isinstance(p1, tuple) and len(p1) == 2 and isinstance(p1[0], (int, float)) and isinstance(p1[1], (int, float))): return float('inf')
    if not (isinstance(p2, tuple) and len(p2) == 2 and isinstance(p2[0], (int, float)) and isinstance(p2[1], (int, float))): return float('inf')
    return float((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def is_inside_box(point: Optional[Tuple[int, int]], box: Optional[Tuple[int, int, int, int]]) -> bool: # Unchanged
    if point is None or box is None: return False; x, y = point; x1, y1, x2, y2 = box; return x1 <= x <= x2 and y1 <= y <= y2
def is_inside_polygon(point: Optional[Tuple[int, int]], polygon: Optional[np.ndarray]) -> bool: # Unchanged
    if point is None: return False
    if polygon is None: return True
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0
def get_keypoints_dict(keypoints_tensor, conf_tensor) -> Dict[str, Dict[str, Any]]: # Unchanged
    kpts = {}
    if keypoints_tensor is None or conf_tensor is None: return kpts
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(keypoints_tensor) and i < len(conf_tensor):
            if len(keypoints_tensor[i]) < 2: continue
            x_val = keypoints_tensor[i][0]; y_val = keypoints_tensor[i][1]; conf_val = conf_tensor[i]
            x_int = int(x_val.item() if hasattr(x_val, 'item') else x_val)
            y_int = int(y_val.item() if hasattr(y_val, 'item') else y_val)
            conf_float = float(conf_val.item() if hasattr(conf_val, 'item') else conf_val)
            if conf_float > 0.3: kpts[name] = {'xy': (x_int, y_int), 'conf': conf_float}
    return kpts
def get_player_color(player_id: str) -> Tuple[int, int, int]: # Unchanged
    if player_id not in player_colors: player_colors[player_id] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
    return player_colors[player_id]
def encode_image(frame: np.ndarray) -> str: # Unchanged
    _, img_encoded = cv2.imencode(".jpg", frame); return base64.b64encode(img_encoded.tobytes()).decode("utf-8")
def calculate_iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float: # Unchanged
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1]); xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]); boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0: return 0.0
    return interArea / (denominator + 1e-6)

# --- Feature Engineering functions (copied from process_pass_data_actual.py) ---
def calculate_distance_feat(p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> float:
    if p1 is None or p2 is None: return 1000.0
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_keypoint_coord_feat(player_keypoints: Dict[str, Any], kp_name: str) -> Optional[Tuple[float, float]]: # player_keypoints is already Dict[str, KeypointData] which has .xy
    kp_data = player_keypoints.get(kp_name) # kp_data is KeypointData object
    if kp_data and hasattr(kp_data, 'xy') and isinstance(kp_data.xy, (list, tuple)) and len(kp_data.xy) == 2:
        try: return tuple(map(float, kp_data.xy))
        except (ValueError, TypeError): return None
    return None

def calculate_speed_feat(
    pos_curr: Optional[Tuple[float, float]],
    pos_prev: Optional[Tuple[float, float]],
    dt: float = FRAME_TIME_DELTA
    ) -> float:
    if pos_curr is None or pos_prev is None or dt == 0: return 0.0
    dist = calculate_distance_feat(pos_curr, pos_prev)
    if dist == 1000.0 : return 0.0
    return dist / dt

def extract_pass_features_for_live_frame(
    current_player_info: Dict[str, Any], # This is a PlayerPoseInfoBase-like dict from consolidated_players
    current_ball_info: Optional[Dict[str, Any]], # This is a BallInfoBase-like dict
    prev_player_frame_data_for_id: Optional[Dict[str, Any]], # Stored data for this player from t-1
    # prev_ball_frame_data_for_id: Optional[Dict[str, Any]] # Stored ball data associated with player's t-1 state
    ) -> List[float]:

    features = []
    num_expected_features = PASS_MODEL_INPUT_SIZE # Use constant

    current_passer_kps = current_player_info.get("keypoints", {}) # keypoints is Dict[str, KeypointData]
    current_player_center = None
    if current_player_info.get("center_display"):
        try: current_player_center = tuple(map(float, current_player_info.get("center_display")))
        except(TypeError, ValueError): current_player_center = None
    
    current_ball_center = None
    if current_ball_info and current_ball_info.get("center"):
        try: current_ball_center = tuple(map(float, current_ball_info.get("center")))
        except(TypeError, ValueError): current_ball_center = None

    prev_passer_kps = {}
    prev_player_center = None
    prev_ball_center = None # Ball associated with player's previous state

    if prev_player_frame_data_for_id:
        prev_player_info_dict = prev_player_frame_data_for_id.get('player_info', {})
        prev_passer_kps = prev_player_info_dict.get("keypoints", {}) # This needs to be Dict[str, KeypointData-like-dict]
        if prev_player_info_dict.get("center_display"):
            try: prev_player_center = tuple(map(float, prev_player_info_dict.get("center_display")))
            except(TypeError, ValueError): prev_player_center = None
        
        prev_ball_info_dict = prev_player_frame_data_for_id.get('ball_info', {})
        if prev_ball_info_dict and prev_ball_info_dict.get("center"):
            try: prev_ball_center = tuple(map(float, prev_ball_info_dict.get("center")))
            except(TypeError, ValueError): prev_ball_center = None

    crk_coord = get_keypoint_coord_feat(current_passer_kps, "right_ankle")
    clk_coord = get_keypoint_coord_feat(current_passer_kps, "left_ankle")
    features.append(calculate_distance_feat(crk_coord, current_ball_center))
    features.append(calculate_distance_feat(clk_coord, current_ball_center))
    features.append(calculate_distance_feat(current_player_center, current_ball_center))
    features.append(calculate_distance_feat(crk_coord, clk_coord))
    cnose_coord = get_keypoint_coord_feat(current_passer_kps, "nose")
    features.append(calculate_distance_feat(cnose_coord, current_ball_center))

    if current_ball_center: features.extend(current_ball_center)
    else: features.extend([0.0, 0.0])
    if crk_coord: features.extend(crk_coord)
    else: features.extend([0.0, 0.0])
    if clk_coord: features.extend(clk_coord)
    else: features.extend([0.0, 0.0])
    
    ball_is_moving_flag = False
    if current_ball_info and isinstance(current_ball_info.get("ball_is_moving"), bool) : # Check if it's explicitly set
        ball_is_moving_flag = current_ball_info.get("ball_is_moving")
    elif current_ball_info: # Fallback if the specific key is missing, but ball is detected
        ball_is_moving_flag = current_ball_info.get("is_moving", False) # Check for an alternative if structure changed

    features.append(1.0 if ball_is_moving_flag else 0.0)


    features.append(calculate_speed_feat(current_player_center, prev_player_center))
    features.append(calculate_speed_feat(current_ball_center, prev_ball_center))
    prk_coord = get_keypoint_coord_feat(prev_passer_kps, "right_ankle") # prev_passer_kps needs to be Dict[str, KeypointData-like-dict]
    features.append(calculate_speed_feat(crk_coord, prk_coord))
    plk_coord = get_keypoint_coord_feat(prev_passer_kps, "left_ankle")
    features.append(calculate_speed_feat(clk_coord, plk_coord))
    
    if len(features) < num_expected_features:
        features.extend([0.0] * (num_expected_features - len(features)))
    return features[:num_expected_features]

# --- Pass Detection LSTM Model Definition (copied from train_pass_detector.py) ---
class PassDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PassDetectorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- FastAPI App and Global Model Initialization ---
app = FastAPI(title="Football AI API - v5_mvp (Pass Detection)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

main_yolo_model: Optional[YOLO] = None
pose_model: Optional[YOLO] = None
aruco_dict: Optional[aruco.Dictionary] = None
aruco_parameters: Optional[aruco.DetectorParameters] = None
ball_tracker_obj: Optional[BallTracker] = None # Renamed to avoid conflict with module
ball_annotator_obj: Optional[BallAnnotator] = None # Renamed
player_pose_smoother_obj: Optional[PlayerPoseSmoother] = None # Renamed
ball_motion_analyzer_obj: Optional[BallMotionAnalyzer] = None # Renamed
ball_kalman_filter_obj: Optional[KalmanFilter2D] = None # Renamed
pass_detection_model: Optional[PassDetectorLSTM] = None
torch_device: Optional[torch.device] = None


@app.on_event("startup")
async def load_models_and_setup():
    global main_yolo_model, pose_model, aruco_dict, aruco_parameters, \
           ball_tracker_obj, ball_annotator_obj, player_pose_smoother_obj, \
           ball_motion_analyzer_obj, ball_kalman_filter_obj, \
           pass_detection_model, torch_device

    print("Attempting to load models and setup (v5_mvp Pass Detection)...")
    try:
        main_yolo_model = YOLO("yolov8n.pt")
        pose_model = YOLO("yolov8n-pose.pt")
    except Exception as e: print(f"FATAL: YOLO models: {e}"); traceback.print_exc(); raise RuntimeError(f"Could not load YOLO: {e}") from e
    print("YOLO Models loaded.")

    # ... (ArUco init - unchanged) ...
    ARUCO_DICT_NAME = "DICT_4X4_100"; print("Initializing ArUco detector...")
    try: aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    except AttributeError:
        try: aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        except Exception as e: print(f"FATAL: ArUco dict: {e}"); traceback.print_exc(); raise RuntimeError(f"ArUco dict: {e}") from e
    if aruco_dict is None: raise RuntimeError("Failed ArUco dictionary")
    try: aruco_parameters = cv2.aruco.DetectorParameters_create()
    except AttributeError: aruco_parameters = aruco.DetectorParameters()
    print("ArUco initialized.")

    ball_tracker_obj = BallTracker(buffer_size=15)
    ball_annotator_obj = BallAnnotator(radius=10, buffer_size=15, thickness=2)
    player_pose_smoother_obj = PlayerPoseSmoother()
    ball_motion_analyzer_obj = BallMotionAnalyzer(history=10, threshold=2.0)
    ball_kalman_filter_obj = KalmanFilter2D()
    print("Trackers, Annotators, Kalman Filters initialized.")

    # Load Pass Detection Model
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for Pass Detection Model: {torch_device}")
    try:
        pass_detection_model = PassDetectorLSTM(PASS_MODEL_INPUT_SIZE, PASS_MODEL_HIDDEN_SIZE, PASS_MODEL_NUM_LAYERS, 1).to(torch_device)
        if Path(PASS_MODEL_PATH).exists():
            pass_detection_model.load_state_dict(torch.load(PASS_MODEL_PATH, map_location=torch_device))
            pass_detection_model.eval()
            print(f"Pass Detection LSTM model loaded successfully from {PASS_MODEL_PATH}")
        else:
            print(f"WARNING: Pass Detection Model weights not found at {PASS_MODEL_PATH}. Model will not be used or will use random weights.")
            pass_detection_model = None # Explicitly set to None if weights not found
    except Exception as e:
        print(f"ERROR loading Pass Detection Model: {e}"); traceback.print_exc()
        pass_detection_model = None # Ensure it's None on error

    print("--- Startup complete (v5_mvp Pass Detection) ---")

# --- Pydantic Models ---
class KeypointData(BaseModel): # Already defined
    xy: Tuple[int, int]; conf: float
class PlayerPoseInfoBase(BaseModel): # Already defined
    player_id: str; box_display: Tuple[int, int, int, int]; center_display: Tuple[int, int]
    keypoints: Dict[str, KeypointData]; type: str
class BallInfoBase(BaseModel): # Already defined
    box: Tuple[int, int, int, int]; center: Tuple[int, int]; conf: float

class DetectedAction(BaseModel):
    action: str
    confidence: float
    player_id: Optional[str] = None # Who performed the action

class FrameAnalysisResult(BaseModel): # Modified
    frame_number_processed: int
    video_filename_processed: str
    players: List[PlayerPoseInfoBase] = Field(default_factory=list)
    ball: Optional[BallInfoBase] = None
    possessor_id: Optional[str] = None
    annotated_image_base64: Optional[str] = None
    num_consolidated_players: int = 0
    num_raw_poses_in_frame: int = 0
    num_players_with_pose_and_id: int = 0
    performance_feedback: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    ball_is_moving: Optional[bool] = None
    detected_actions: List[DetectedAction] = Field(default_factory=list) # New field for actions

# (FrameDataForStorage model can remain the same if you don't save detected_actions to per-frame JSON)

# --- Main Processing Function ---
def process_frame_for_analysis(
    frame: np.ndarray,
    frame_number_processed: int,
    video_filename_processed: str,
    playing_field_polygon_coords: Optional[List[Tuple[int,int]]] = None,
    return_annotated_image: bool = True,
    debug_print_interval: int = 30
) -> FrameAnalysisResult:
    # ... (Initial checks and setup, PLAYING_FIELD_POLYGON, debug_print, annotated_frame init - same as before) ...
    global player_feature_buffers, player_previous_frame_data # Access global buffers

    if not all([main_yolo_model, pose_model, aruco_dict, aruco_parameters,
                ball_tracker_obj, ball_annotator_obj, player_pose_smoother_obj,
                ball_motion_analyzer_obj, ball_kalman_filter_obj]): # pass_detection_model can be None
        print("CRITICAL ERROR: Core models/helpers NOT INITIALIZED!")
        raise HTTPException(status_code=503, detail="Core models not loaded.")

    current_frame_player_tracking_info_for_drawing = {}
    frame_height, frame_width = frame.shape[:2]
    if playing_field_polygon_coords: PLAYING_FIELD_POLYGON = np.array(playing_field_polygon_coords, np.int32)
    else: PLAYING_FIELD_POLYGON = np.array([[0,0],[frame_width-1,0],[frame_width-1,frame_height-1],[0,frame_height-1]], np.int32)
    debug_print = (frame_number_processed % debug_print_interval == 0)
    if debug_print: print(f"\n--- FRAME {frame_number_processed} ('{video_filename_processed}') ---")
    annotated_frame = frame.copy() if return_annotated_image else frame
    if return_annotated_image and PLAYING_FIELD_POLYGON is not None:
        cv2.polylines(annotated_frame, [PLAYING_FIELD_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

    # --- YOLO Person Tracking (same as before) ---
    yolo_person_detections_with_ids = []
    try:
        yolo_person_results = main_yolo_model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)[0]
        raw_tracker_ids = None
        if hasattr(yolo_person_results, 'boxes') and yolo_person_results.boxes is not None and hasattr(yolo_person_results.boxes, 'id') and yolo_person_results.boxes.id is not None:
            raw_tracker_ids = yolo_person_results.boxes.id.int().cpu().tolist()
        if yolo_person_results.boxes is not None:
            for i in range(len(yolo_person_results.boxes)):
                box_data = yolo_person_results.boxes[i]
                x1,y1,x2,y2 = map(int,box_data.xyxy[0]); center = get_center((x1,y1,x2,y2))
                if not is_inside_polygon(center, PLAYING_FIELD_POLYGON): continue
                yolo_tracker_id = raw_tracker_ids[i] if raw_tracker_ids and i < len(raw_tracker_ids) else None
                conf = float(box_data.conf[0].item()) if hasattr(box_data, 'conf') and box_data.conf is not None else 0.0
                yolo_person_detections_with_ids.append({'box_person':(x1,y1,x2,y2), 'center_person':center,'yolo_tracker_id':yolo_tracker_id,'conf':conf})
    except Exception as e_track:
        if debug_print: print(f"  WARNING: YOLO person tracking FAILED: {e_track}"); traceback.print_exc()
    
    # --- ArUco Detection (same as before) ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    detected_aruco_players = {} # ... (populate as before) ...
    if ids is not None:
        if return_annotated_image: aruco.drawDetectedMarkers(annotated_frame, corners, ids)
        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0]); marker_corners = corners[i]
            x_coords,y_coords = marker_corners[0,:,0], marker_corners[0,:,1]
            aruco_box=(int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))
            if not is_inside_polygon(get_center(aruco_box), PLAYING_FIELD_POLYGON): continue
            detected_aruco_players[marker_id] = {'box_aruco': aruco_box, 'center_aruco': get_aruco_center(marker_corners)}

    # --- Player Consolidation (same as before, populates consolidated_players_this_frame) ---
    consolidated_players_this_frame_dicts: List[Dict[str, Any]] = [] # List of dicts
    player_id_to_color_map = {}
    # ... (ArUco consolidation) ...
    for aruco_id, aruco_data in detected_aruco_players.items():
        player_id_str=f"A-{aruco_id}"; best_yolo_match_box=aruco_data['box_aruco']
        min_dist_yolo = float('inf')
        for yolo_person in yolo_person_detections_with_ids:
            if is_inside_box(aruco_data['center_aruco'], yolo_person['box_person']):
                dist = get_distance_sq(aruco_data['center_aruco'], yolo_person['center_person'])
                if dist < min_dist_yolo: min_dist_yolo = dist; best_yolo_match_box = yolo_person['box_person']
        p_info={'player_id':player_id_str,'box_display':best_yolo_match_box,'center_display':get_center(best_yolo_match_box),'type':'aruco'}
        consolidated_players_this_frame_dicts.append(p_info)
        player_id_to_color_map[player_id_str] = get_player_color(player_id_str)
        current_frame_player_tracking_info_for_drawing[player_id_str] = {'current_box':best_yolo_match_box, 'color': player_id_to_color_map[player_id_str]}
    # ... (YOLO-only consolidation) ...
    for yolo_person in yolo_person_detections_with_ids:
        if yolo_person['yolo_tracker_id'] is not None:
            is_already_consolidated_as_aruco = False
            for p_info in consolidated_players_this_frame_dicts:
                if p_info['type'] == 'aruco' and calculate_iou(yolo_person['box_person'], p_info['box_display']) > 0.5:
                    is_already_consolidated_as_aruco = True; break
            if not is_already_consolidated_as_aruco:
                player_id_str = f"Y-{yolo_person['yolo_tracker_id']}"
                if not any(p['player_id'] == player_id_str for p in consolidated_players_this_frame_dicts):
                    p_info = {'player_id': player_id_str, 'box_display': yolo_person['box_person'],
                              'center_display': yolo_person['center_person'], 'type': 'yolo'}
                    consolidated_players_this_frame_dicts.append(p_info)
                    player_id_to_color_map[player_id_str] = get_player_color(player_id_str)
                    current_frame_player_tracking_info_for_drawing[player_id_str] = {'current_box': yolo_person['box_person'], 'color': player_id_to_color_map[player_id_str]}

    # --- Ball Detection (same as before, using main_yolo_model) ---
    ball_result_for_storage: Optional[BallInfoBase] = None; ball_moving_status: Optional[bool] = None
    current_ball_info_dict_for_features: Optional[Dict[str, Any]] = None # For feature extraction
    yolo_ball_results_raw = main_yolo_model(frame, classes=[32], verbose=False)[0]
    # ... (populate ball_result_for_storage and ball_moving_status) ...
    raw_ball_xyxy,raw_ball_conf,raw_ball_class_id = [],[],[]
    if yolo_ball_results_raw.boxes is not None:
        for box_data in yolo_ball_results_raw.boxes:
            x1,y1,x2,y2=map(int,box_data.xyxy[0]); center=get_center((x1,y1,x2,y2)); conf=float(box_data.conf[0].item()) if hasattr(box_data, 'conf') and box_data.conf is not None else 0.0
            if conf > 0.35 and is_inside_polygon(center, PLAYING_FIELD_POLYGON):
                raw_ball_xyxy.append([x1,y1,x2,y2]); raw_ball_conf.append(conf); raw_ball_class_id.append(32)
    candidate_sv_detections = sv.Detections.empty();
    if raw_ball_xyxy: candidate_sv_detections = sv.Detections(xyxy=np.array(raw_ball_xyxy),confidence=np.array(raw_ball_conf),class_id=np.array(raw_ball_class_id))
    tracked_ball_raw_sv_detections = ball_tracker_obj.update(candidate_sv_detections) # Use renamed object
    if len(tracked_ball_raw_sv_detections) > 0:
        raw_ball_data = tracked_ball_raw_sv_detections[0]
        b_raw_box = tuple(map(int,raw_ball_data.xyxy[0])); b_raw_conf = float(raw_ball_data.confidence[0]) if raw_ball_data.confidence is not None and len(raw_ball_data.confidence)>0 else 0.5
        b_raw_center = get_center(b_raw_box)
        ball_kalman_filter_obj.correct(b_raw_center); b_smoothed_center = ball_kalman_filter_obj.predict() # Use renamed object
        ball_result_for_storage = BallInfoBase(box=b_raw_box, center=b_smoothed_center, conf=b_raw_conf)
        current_ball_info_dict_for_features = {"center": b_smoothed_center, "box": b_raw_box, "conf": b_raw_conf} # For feature extraction
        if ball_motion_analyzer_obj:  # Use renamed object
            ball_motion_analyzer_obj.update(b_smoothed_center)
            ball_moving_status = ball_motion_analyzer_obj.is_moving()
            if current_ball_info_dict_for_features: current_ball_info_dict_for_features["ball_is_moving"] = ball_moving_status
        if return_annotated_image and ball_annotator_obj: # Use renamed object
            annotated_frame = ball_annotator_obj.annotate(annotated_frame, tracked_ball_raw_sv_detections)
            cv2.circle(annotated_frame, b_smoothed_center, 5, (0,0,255), -1)
            if ball_moving_status is not None: cv2.putText(annotated_frame, "Ball Moving" if ball_moving_status else "Ball Stationary", (frame_width-220,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    # --- Pose Estimation (same as before) ---
    pose_results = pose_model(frame, verbose=False)[0]; all_raw_poses_in_frame = []
    # ... (populate all_raw_poses_in_frame) ...
    if pose_results.keypoints is not None and pose_results.boxes is not None:
        # ... (logic from v5.4.1 to populate all_raw_poses_in_frame)
        keypoints_data=pose_results.keypoints.xy; conf_data=pose_results.keypoints.conf; boxes_data=pose_results.boxes.xyxy
        num_poses_detected = keypoints_data.shape[0]
        if num_poses_detected == boxes_data.shape[0] and (conf_data is None or num_poses_detected == conf_data.shape[0]):
            for i in range(num_poses_detected):
                current_conf_tensor = conf_data[i] if conf_data is not None else np.zeros(keypoints_data.shape[1])
                raw_kpts_dict=get_keypoints_dict(keypoints_data[i], current_conf_tensor)
                pose_bbox=tuple(map(int,boxes_data[i]))
                if raw_kpts_dict and is_inside_polygon(get_center(pose_bbox),PLAYING_FIELD_POLYGON):
                     all_raw_poses_in_frame.append({'keypoints':raw_kpts_dict,'box':pose_bbox})


    # --- Pose Association & Smoothing (produces players_with_pose_for_storage) ---
    players_with_pose_for_storage: List[PlayerPoseInfoBase] = [] # This will be the output list
    # This loop iterates through consolidated_players_this_frame_dicts
    for p_info_consolidated_dict in consolidated_players_this_frame_dicts:
        player_id_str=p_info_consolidated_dict['player_id']
        player_consolidated_box=p_info_consolidated_dict['box_display']
        best_match_raw_pose_kpts_dict=None; best_match_iou=0.0
        MIN_IOU_FOR_POSE_ASSOCIATION = 0.1 # Keep this from previous successful version

        for i, raw_pose_info in enumerate(all_raw_poses_in_frame):
            current_raw_kpts=raw_pose_info['keypoints']; current_pose_box=raw_pose_info['box']
            iou=calculate_iou(player_consolidated_box,current_pose_box)
            if iou > best_match_iou and iou > MIN_IOU_FOR_POSE_ASSOCIATION:
                best_match_iou=iou; best_match_raw_pose_kpts_dict=current_raw_kpts
        
        if best_match_raw_pose_kpts_dict and player_pose_smoother_obj:
            smoothed_kpts_dict=player_pose_smoother_obj.smooth(player_id_str,best_match_raw_pose_kpts_dict) # Use renamed
            pydantic_kpts={name:KeypointData(**data) for name,data in smoothed_kpts_dict.items()}
            
            # Create the PlayerPoseInfoBase object to be stored/returned
            player_pose_base_item = PlayerPoseInfoBase(
                player_id=player_id_str, 
                box_display=p_info_consolidated_dict['box_display'], 
                center_display=p_info_consolidated_dict['center_display'],
                keypoints=pydantic_kpts, # These are smoothed keypoints
                type=p_info_consolidated_dict['type']
            )
            players_with_pose_for_storage.append(player_pose_base_item)
            
            # Store smoothed keypoints also for drawing and feature extraction this frame
            if player_id_str in current_frame_player_tracking_info_for_drawing:
                 current_frame_player_tracking_info_for_drawing[player_id_str]['current_pose_smoothed'] = smoothed_kpts_dict
                 # Store for feature extraction: current player's full Pydantic model for consistency
                 current_frame_player_tracking_info_for_drawing[player_id_str]['player_pose_info_base'] = player_pose_base_item


    # --- Pass Detection using LSTM ---
    detected_actions_output: List[DetectedAction] = []
    if pass_detection_model is not None and torch_device is not None:
        for player_id, player_draw_info in current_frame_player_tracking_info_for_drawing.items():
            if 'player_pose_info_base' not in player_draw_info: # Needs pose to extract features
                continue

            current_player_data_for_features = player_draw_info['player_pose_info_base'] # This is PlayerPoseInfoBase
            
            # Get previous frame data for this specific player
            prev_data_for_this_player = player_previous_frame_data.get(player_id)
            prev_player_info_for_features = prev_data_for_this_player.get('player_info') if prev_data_for_this_player else None
            prev_ball_info_for_features = prev_data_for_this_player.get('ball_info') if prev_data_for_this_player else None

            # Extract features for the current frame
            # We need to pass KeypointData objects to get_keypoint_coord_feat
            # current_player_data_for_features.keypoints is Dict[str, KeypointData]
            live_features = extract_pass_features_for_live_frame(
                current_player_data_for_features.model_dump(), # Pass as dict
                current_ball_info_dict_for_features, # Pass current ball info as dict
                prev_player_info_for_features, # This is already a dict
                prev_ball_info_for_features    # This is already a dict
            )

            player_feature_buffers[player_id].append(live_features)

            # Store current data for next frame's "previous"
            # We need to store the dict representation of PlayerPoseInfoBase and BallInfoBase
            player_previous_frame_data[player_id] = {
                'player_info': current_player_data_for_features.model_dump(), # Store as dict
                'ball_info': current_ball_info_dict_for_features # Store current ball info
            }
            
            if len(player_feature_buffers[player_id]) == PASS_MODEL_SEQUENCE_LENGTH:
                sequence_to_predict = list(player_feature_buffers[player_id])
                sequence_tensor = torch.tensor([sequence_to_predict], dtype=torch.float32).to(torch_device) # Batch size of 1

                with torch.no_grad():
                    output_logits = pass_detection_model(sequence_tensor)
                    probability = torch.sigmoid(output_logits).item()

                if debug_print: print(f"    Pass detection for player {player_id}: Prob={probability:.4f}")

                if probability > PASS_DETECTION_THRESHOLD:
                    detected_actions_output.append(DetectedAction(action="Pass", confidence=probability, player_id=player_id))
                    if return_annotated_image:
                        p_box = player_draw_info['current_box']
                        cv2.putText(annotated_frame, f"PASS DETECTED ({probability:.2f})", 
                                    (p_box[0], p_box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2)
    
    # --- Clean up buffers for players no longer tracked ---
    active_player_ids = set(current_frame_player_tracking_info_for_drawing.keys())
    for player_id in list(player_feature_buffers.keys()): # Iterate over copy of keys
        if player_id not in active_player_ids:
            del player_feature_buffers[player_id]
            if player_id in player_previous_frame_data:
                del player_previous_frame_data[player_id]


    # --- Skeleton Drawing (same as before) ---
    if return_annotated_image: # ... (copy skeleton drawing logic) ...
        for p_draw_info in current_frame_player_tracking_info_for_drawing.values():
            smoothed_kpts_for_drawing = p_draw_info.get('current_pose_smoothed')
            if smoothed_kpts_for_drawing:
                p_color = p_draw_info['color']
                for i_skel,j_skel in SKELETON_CONNECTIONS:
                    kp1_name = KEYPOINT_NAMES[i_skel]; kp2_name = KEYPOINT_NAMES[j_skel]
                    kp1_info=smoothed_kpts_for_drawing.get(kp1_name); kp2_info=smoothed_kpts_for_drawing.get(kp2_name)
                    if kp1_info and kp1_info.get('xy') and kp2_info and kp2_info.get('xy'):
                        cv2.line(annotated_frame,kp1_info['xy'],kp2_info['xy'],p_color,2)
                for kp_data_draw in smoothed_kpts_for_drawing.values():
                    if kp_data_draw.get('xy'): cv2.circle(annotated_frame, kp_data_draw['xy'], 3, p_color, -1)

    # --- Possession Detection (same as before, uses players_with_pose_for_storage) ---
    possessor_final_id: Optional[str] = None # ... (populate as before) ...
    if ball_result_for_storage and ball_result_for_storage.center:
        # ... (possession logic as in v5.4.x, using ball_result_for_storage.center
        # and players_with_pose_for_storage which contains PlayerPoseInfoBase objects)
        ball_center_smoothed=ball_result_for_storage.center; min_ball_dist_sq=float('inf'); POSSESSION_THRESHOLD_SQ=4500
        for player_with_pose in players_with_pose_for_storage:
            smoothed_kpts={k:{'xy':v.xy,'conf':v.conf} for k,v in player_with_pose.keypoints.items()}
            relevant_kpt_names=["left_ankle","right_ankle","left_knee","right_knee","left_wrist","right_wrist"]
            closest_player_part_dist_sq=float('inf')
            for kp_name in relevant_kpt_names:
                if kp_name in smoothed_kpts and smoothed_kpts[kp_name].get('xy') is not None:
                    part_xy = smoothed_kpts[kp_name]['xy']
                    dist_sq = get_distance_sq(part_xy, ball_center_smoothed)
                    if isinstance(dist_sq, (int, float)) and isinstance(closest_player_part_dist_sq, (int, float)):
                        if dist_sq < closest_player_part_dist_sq: closest_player_part_dist_sq=dist_sq
            if isinstance(closest_player_part_dist_sq, (int, float)) and closest_player_part_dist_sq < POSSESSION_THRESHOLD_SQ:
                if closest_player_part_dist_sq < min_ball_dist_sq:
                    min_ball_dist_sq = closest_player_part_dist_sq; possessor_final_id = player_with_pose.player_id


    if return_annotated_image and possessor_final_id and possessor_final_id in current_frame_player_tracking_info_for_drawing:
        # ... (draw possession box) ...
        p_draw_info = current_frame_player_tracking_info_for_drawing[possessor_final_id]
        px1,py1,px2,py2 = p_draw_info['current_box']
        cv2.rectangle(annotated_frame,(px1,py1),(px2,py2),(0,0,255),3)
        cv2.putText(annotated_frame,f"ID {possessor_final_id} Has Ball",(px1,py1-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)


    annotated_image_b64_str = encode_image(annotated_frame) if return_annotated_image else None

    return FrameAnalysisResult(
        frame_number_processed=frame_number_processed, video_filename_processed=video_filename_processed,
        players=players_with_pose_for_storage, ball=ball_result_for_storage,
        possessor_id=possessor_final_id, annotated_image_base64=annotated_image_b64_str,
        num_consolidated_players=len(consolidated_players_this_frame_dicts), # Use the dict list for count
        num_raw_poses_in_frame=len(all_raw_poses_in_frame),
        num_players_with_pose_and_id=len(players_with_pose_for_storage),
        ball_is_moving=ball_moving_status,
        detected_actions=detected_actions_output # Add detected actions
    )

# --- FastAPI Endpoint (largely unchanged, ensure it calls the updated process_frame_for_analysis) ---
@app.post("/process_frame/", response_model=FrameAnalysisResult)
async def analyze_frame_endpoint(
    image_file: UploadFile = File(...),
    playing_field_polygon_coords_json: Optional[str] = Form(None),
    return_annotated_image: bool = Form(True),
    save_to_json_dir: Optional[str] = Form(None),
    video_filename: str = Form("unknown_video"),
    frame_number: int = Form(...)
):
    request_time_start = time.time()
    req_id_suffix = int(request_time_start*1000)%10000
    # if frame_number % 10 == 0: print(f"Req-{req_id_suffix}: Frame {frame_number} ...") # Reduced verbosity

    contents = await image_file.read()
    frame_cv = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame_cv is None: raise HTTPException(status_code=400, detail="Could not decode image.")

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
        analysis_output: FrameAnalysisResult = process_frame_for_analysis(
            frame_cv, frame_number_processed=frame_number, video_filename_processed=video_filename,
            playing_field_polygon_coords=polygon_coords_list, return_annotated_image=return_annotated_image
        )
        # Note: performance_feedback is already handled by default_factory in Pydantic model
        # If you want to re-enable Get_Perf_From_Pose, ensure it can handle PlayerPoseInfoBase or reconstruct PlayerPoseInfo

        if save_to_json_dir: # ... (JSON saving logic - unchanged from v5.4.x conceptually) ...
            try:
                sane_video_filename = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in video_filename).rstrip()
                if not sane_video_filename: sane_video_filename = "unknown_video"
                video_specific_dir = Path(save_to_json_dir) / sane_video_filename
                video_specific_dir.mkdir(parents=True, exist_ok=True)
                json_filename = f"frame_{frame_number:06d}.json"
                json_filepath = video_specific_dir / json_filename
                # FrameDataForStorage model would need to be updated if you want to save detected_actions
                # For now, assuming FrameDataForStorage remains as is for lean per-frame raw data.
                storage_data_dict = {
                    "frame_number_processed": analysis_output.frame_number_processed,
                    "video_filename_processed": analysis_output.video_filename_processed,
                    "players": [p.model_dump() for p in analysis_output.players],
                    "ball": analysis_output.ball.model_dump() if analysis_output.ball else None,
                    "possessor_id": analysis_output.possessor_id,
                    "num_consolidated_players": analysis_output.num_consolidated_players,
                    "num_raw_poses_in_frame": analysis_output.num_raw_poses_in_frame,
                    "num_players_with_pose_and_id": analysis_output.num_players_with_pose_and_id,
                    "ball_is_moving": analysis_output.ball_is_moving,
                    # "detected_actions": [a.model_dump() for a in analysis_output.detected_actions] # Optionally save actions
                }
                with open(json_filepath, 'w') as f: json.dump(storage_data_dict, f, indent=2)
                # if frame_number % 10 == 0: print(f"Req-{req_id_suffix}: ✓ Saved JSON for frame {frame_number}")
            except Exception as e_save:
                print(f"Req-{req_id_suffix}: !!! ERROR saving JSON for frame {frame_number}: {e_save} !!!"); traceback.print_exc()


        processing_time_end = time.time()
        if frame_number % 30 == 0: # Print less frequently
             print(f"Req-{req_id_suffix}: Frame {frame_number} processed in {processing_time_end - request_time_start:.4f} secs. Detected actions: {len(analysis_output.detected_actions)}")
        return analysis_output
    # ... (Exception handling - unchanged) ...
    except HTTPException as e: print(f"Req-{req_id_suffix}: HTTP Ex: {e.detail}"); raise e
    except RuntimeError as e: print(f"Req-{req_id_suffix}: Runtime err: {e}"); traceback.print_exc(); raise HTTPException(status_code=503, detail=str(e))
    except Exception as e: print(f"Req-{req_id_suffix}: !!! UNEXPECTED err: {e} !!!"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn server for v5_mvp_pass_detection.py ...")
    # Ensure PASS_MODEL_PATH points to your actual trained .pth file
    # Ensure bytetrack.yaml is in the working directory.
    uvicorn.run("v5_mvp_pass_detection:app", host="0.0.0.0", port=8000, workers=1) # Ensure filename matches