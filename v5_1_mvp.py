# v5_mvp.py (Formerly v5.4.5_mvp.py - Kalman & Drawing Refinement)
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
from collections import deque
import supervision as sv
import traceback

# Import from custom modules
from feedback import Get_Perf_From_Pose
from pose_ball_tracking_kalman import PlayerPoseSmoother, BallMotionAnalyzer, KalmanFilter2D # Uses updated version

# --- Constants ---
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

# --- Ball Tracker & Annotator Classes ---
class BallAnnotator: # Unchanged
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

class BallTracker: # Unchanged
    def __init__(self, buffer_size: int = 10): self.buffer = deque(maxlen=buffer_size)
    def update(self, detections: sv.Detections) -> sv.Detections:
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

# --- Helper Functions ---
def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]: # Unchanged
    x1, y1, x2, y2 = bbox; return (int((x1 + x2) / 2), int((y1 + y2) / 2))
def get_aruco_center(corners) -> Tuple[int, int]: # Unchanged
    return (int(np.mean(corners[0,:,0])), int(np.mean(corners[0,:,1])))
def get_distance_sq(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> float: # Unchanged from previous fix
    if p1 is None or p2 is None: return float('inf')
    if not (isinstance(p1, tuple) and len(p1) == 2 and isinstance(p1[0], (int, float)) and isinstance(p1[1], (int, float))):
        print(f"CRITICAL WARNING: get_distance_sq received invalid p1: {p1} (type {type(p1)})")
        return float('inf')
    if not (isinstance(p2, tuple) and len(p2) == 2 and isinstance(p2[0], (int, float)) and isinstance(p2[1], (int, float))):
        print(f"CRITICAL WARNING: get_distance_sq received invalid p2: {p2} (type {type(p2)})")
        return float('inf')
    dist_sq = float((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist_sq
def is_inside_box(point: Optional[Tuple[int, int]], box: Optional[Tuple[int, int, int, int]]) -> bool: # Unchanged
    if point is None or box is None: return False; x, y = point; x1, y1, x2, y2 = box; return x1 <= x <= x2 and y1 <= y <= y2
def is_inside_polygon(point: Optional[Tuple[int, int]], polygon: Optional[np.ndarray]) -> bool: # Unchanged
    if point is None: return False
    if polygon is None: return True
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0
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

# --- FastAPI App and Global Model Initialization ---
app = FastAPI(title="Football AI API - v5_mvp (Kalman & Drawing Refinement)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

main_yolo_model: Optional[YOLO] = None
pose_model: Optional[YOLO] = None
aruco_dict: Optional[aruco.Dictionary] = None
aruco_parameters: Optional[aruco.DetectorParameters] = None
ball_tracker: Optional[BallTracker] = None
ball_annotator: Optional[BallAnnotator] = None
player_pose_smoother: Optional[PlayerPoseSmoother] = None
ball_motion_analyzer: Optional[BallMotionAnalyzer] = None
ball_kalman_filter: Optional[KalmanFilter2D] = None

@app.on_event("startup") # Unchanged
async def load_models_and_setup():
    global main_yolo_model, pose_model, aruco_dict, aruco_parameters, \
           ball_tracker, ball_annotator, player_pose_smoother, \
           ball_motion_analyzer, ball_kalman_filter
    print("Attempting to load models and setup (v5_mvp Kalman & Drawing Refinement)...")
    try:
        main_yolo_model = YOLO("yolov8n.pt")
        pose_model = YOLO("yolov8n-pose.pt")
        print(f"Main YOLO model: {type(main_yolo_model)}, Pose model: {type(pose_model)}")
    except Exception as e:
        print(f"FATAL: Error loading YOLO models: {e}"); traceback.print_exc(); raise RuntimeError(f"Could not load YOLO models: {e}") from e
    print("YOLO Models loaded successfully.")
    print("Initializing ArUco detector...")
    ARUCO_DICT_NAME = "DICT_4X4_100"
    try: aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    except AttributeError:
        try: aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        except Exception as e: print(f"FATAL: ArUco dict init failed: {e}"); traceback.print_exc(); raise RuntimeError(f"Could not init ArUco dict: {e}") from e
    if aruco_dict is None: print("FATAL: ArUco dict is None after attempts"); raise RuntimeError("Failed to load ArUco dictionary")
    try: aruco_parameters = cv2.aruco.DetectorParameters_create(); print("Using cv2.aruco.DetectorParameters_create()")
    except AttributeError: aruco_parameters = aruco.DetectorParameters(); print("Using cv2.aruco.DetectorParameters() as fallback")
    print("ArUco detector initialized.")
    print("Initializing Ball Tracker, Annotator, and Kalman Filters...")
    ball_tracker = BallTracker(buffer_size=15); ball_annotator = BallAnnotator(radius=10, buffer_size=15, thickness=2)
    player_pose_smoother = PlayerPoseSmoother(); ball_motion_analyzer = BallMotionAnalyzer(history=10, threshold=2.0)
    ball_kalman_filter = KalmanFilter2D()
    print("Ball Tracker, Annotator, and Kalman Filters initialized successfully.")
    print("--- Startup complete (v5_mvp Kalman & Drawing Refinement) ---")

# --- Pydantic Models --- # Unchanged
class KeypointData(BaseModel):
    xy: Tuple[int, int]; conf: float
class PlayerPoseInfoBase(BaseModel):
    player_id: str; box_display: Tuple[int, int, int, int]; center_display: Tuple[int, int]
    keypoints: Dict[str, KeypointData]; type: str
class BallInfoBase(BaseModel):
    box: Tuple[int, int, int, int]; center: Tuple[int, int]; conf: float
class FrameDataForStorage(BaseModel):
    frame_number_processed: int
    video_filename_processed: str
    players: List[PlayerPoseInfoBase] = Field(default_factory=list)
    ball: Optional[BallInfoBase] = None
    possessor_id: Optional[str] = None
    num_consolidated_players: int = 0
    num_raw_poses_in_frame: int = 0
    num_players_with_pose_and_id: int = 0
    ball_is_moving: Optional[bool] = None
class FrameAnalysisResult(BaseModel):
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

# --- Main Processing Function ---
def process_frame_for_analysis(
    frame: np.ndarray,
    frame_number_processed: int,
    video_filename_processed: str,
    playing_field_polygon_coords: Optional[List[Tuple[int,int]]] = None,
    return_annotated_image: bool = True,
    debug_print_interval: int = 10 # Can increase this interval once stable
) -> FrameAnalysisResult:

    if not all([main_yolo_model, pose_model, aruco_dict, aruco_parameters,
                ball_tracker, ball_annotator, player_pose_smoother,
                ball_motion_analyzer, ball_kalman_filter]):
        print("CRITICAL ERROR: One or more core models/helpers are NOT INITIALIZED in process_frame_for_analysis!")
        raise HTTPException(status_code=503, detail="Core models or helpers not loaded/initialized.")

    current_frame_player_tracking_info_for_drawing = {}
    frame_height, frame_width = frame.shape[:2]
    if playing_field_polygon_coords:
        PLAYING_FIELD_POLYGON = np.array(playing_field_polygon_coords, np.int32)
    else:
        PLAYING_FIELD_POLYGON = np.array(
            [[0,0],[frame_width-1,0],[frame_width-1,frame_height-1],[0,frame_height-1]], np.int32)

    debug_print = (frame_number_processed % debug_print_interval == 0)
    if debug_print:
        print(f"\n--- FRAME {frame_number_processed} ('{video_filename_processed}') ---")
        # print(f"  Using PLAYING_FIELD_POLYGON: {PLAYING_FIELD_POLYGON.tolist() if PLAYING_FIELD_POLYGON is not None else 'None (Full Frame)'}")

    annotated_frame = frame.copy() if return_annotated_image else frame

    if return_annotated_image and PLAYING_FIELD_POLYGON is not None:
        cv2.polylines(annotated_frame, [PLAYING_FIELD_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

    yolo_person_detections_with_ids = []
    try:
        # if debug_print: print(f"  Calling main_yolo_model.track() with persist=True, tracker='bytetrack.yaml'")
        yolo_person_results = main_yolo_model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)[0]
        raw_tracker_ids = None
        if hasattr(yolo_person_results, 'boxes') and yolo_person_results.boxes is not None and \
           hasattr(yolo_person_results.boxes, 'id') and yolo_person_results.boxes.id is not None:
            raw_tracker_ids = yolo_person_results.boxes.id.int().cpu().tolist()
            # if debug_print: print(f"  RAW Tracker IDs from YOLO: {raw_tracker_ids}")
        # elif debug_print: print(f"  YOLO track call did NOT return IDs")

        if yolo_person_results.boxes is not None:
            for i in range(len(yolo_person_results.boxes)):
                box_data = yolo_person_results.boxes[i]
                x1,y1,x2,y2 = map(int,box_data.xyxy[0]); center = get_center((x1,y1,x2,y2))
                is_inside = is_inside_polygon(center, PLAYING_FIELD_POLYGON)
                if not is_inside:
                    # if debug_print: print(f"      Person at {center} (box [{x1},{y1},{x2},{y2}]) is OUTSIDE polygon. Skipping.")
                    continue
                yolo_tracker_id = None
                if raw_tracker_ids and i < len(raw_tracker_ids): yolo_tracker_id = raw_tracker_ids[i]
                # elif debug_print : print(f"      WARNING: No raw_tracker_id for box index {i}")
                conf = float(box_data.conf[0].item()) if hasattr(box_data, 'conf') and box_data.conf is not None else 0.0
                yolo_person_detections_with_ids.append({'box_person':(x1,y1,x2,y2), 'center_person':center,'yolo_tracker_id':yolo_tracker_id,'conf':conf})
        # if debug_print:
            # print(f"  Processed YOLO Detections (persons) after polygon filter: {len(yolo_person_detections_with_ids)}")
            # if yolo_person_detections_with_ids: print(f"    Processed Tracker IDs: {[p['yolo_tracker_id'] for p in yolo_person_detections_with_ids]}")
    except Exception as e_track:
        if debug_print: print(f"  CRITICAL WARNING: YOLO person tracking FAILED: {e_track}"); traceback.print_exc()
        yolo_person_detections_with_ids = []

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    detected_aruco_players = {}
    if ids is not None:
        if return_annotated_image: aruco.drawDetectedMarkers(annotated_frame, corners, ids)
        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0]); marker_corners = corners[i]
            x_coords,y_coords = marker_corners[0,:,0], marker_corners[0,:,1]
            aruco_box=(int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))
            if not is_inside_polygon(get_center(aruco_box), PLAYING_FIELD_POLYGON): continue
            detected_aruco_players[marker_id] = {'box_aruco': aruco_box, 'center_aruco': get_aruco_center(marker_corners)}
    # if debug_print: print(f"  Detected ArUco markers: {len(detected_aruco_players)}")

    consolidated_players_this_frame = []
    player_id_to_color_map = {}
    for aruco_id, aruco_data in detected_aruco_players.items():
        player_id_str=f"A-{aruco_id}"; best_yolo_match_box=aruco_data['box_aruco']
        min_dist_yolo = float('inf')
        for yolo_person in yolo_person_detections_with_ids:
            if is_inside_box(aruco_data['center_aruco'], yolo_person['box_person']):
                dist = get_distance_sq(aruco_data['center_aruco'], yolo_person['center_person'])
                if dist < min_dist_yolo: min_dist_yolo = dist; best_yolo_match_box = yolo_person['box_person']
        p_info={'player_id':player_id_str,'box_display':best_yolo_match_box,'center_display':get_center(best_yolo_match_box),'type':'aruco'}
        consolidated_players_this_frame.append(p_info)
        player_id_to_color_map[player_id_str] = get_player_color(player_id_str)
        current_frame_player_tracking_info_for_drawing[player_id_str] = {'current_box':best_yolo_match_box, 'color': player_id_to_color_map[player_id_str]}

    for yolo_person in yolo_person_detections_with_ids:
        if yolo_person['yolo_tracker_id'] is not None:
            is_already_consolidated_as_aruco = False
            for p_info in consolidated_players_this_frame:
                if p_info['type'] == 'aruco' and calculate_iou(yolo_person['box_person'], p_info['box_display']) > 0.5:
                    is_already_consolidated_as_aruco = True; break
            if not is_already_consolidated_as_aruco:
                player_id_str = f"Y-{yolo_person['yolo_tracker_id']}"
                if not any(p['player_id'] == player_id_str for p in consolidated_players_this_frame):
                    p_info = {'player_id': player_id_str, 'box_display': yolo_person['box_person'],
                              'center_display': yolo_person['center_person'], 'type': 'yolo'}
                    consolidated_players_this_frame.append(p_info)
                    player_id_to_color_map[player_id_str] = get_player_color(player_id_str)
                    current_frame_player_tracking_info_for_drawing[player_id_str] = {'current_box': yolo_person['box_person'], 'color': player_id_to_color_map[player_id_str]}
    # if debug_print: print(f"  Consolidated players: {len(consolidated_players_this_frame)}. IDs: {[p['player_id'] for p in consolidated_players_this_frame]}")

    if return_annotated_image:
        for p_info in consolidated_players_this_frame:
            color_to_use = player_id_to_color_map.get(p_info['player_id'], (255,0,255))
            b = p_info['box_display']
            cv2.rectangle(annotated_frame, (b[0], b[1]), (b[2], b[3]), color_to_use, 2)
            cv2.putText(annotated_frame, f"ID:{p_info['player_id']}", (b[0], b[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_to_use, 1)

    ball_result_for_storage: Optional[BallInfoBase] = None; ball_moving_status: Optional[bool] = None
    yolo_ball_results_raw = main_yolo_model(frame, classes=[32], verbose=False)[0]
    raw_ball_xyxy,raw_ball_conf,raw_ball_class_id = [],[],[]
    if yolo_ball_results_raw.boxes is not None:
        for box_data in yolo_ball_results_raw.boxes:
            x1,y1,x2,y2=map(int,box_data.xyxy[0]); center=get_center((x1,y1,x2,y2)); conf=float(box_data.conf[0].item()) if hasattr(box_data, 'conf') and box_data.conf is not None else 0.0
            if conf > 0.35 and is_inside_polygon(center, PLAYING_FIELD_POLYGON):
                raw_ball_xyxy.append([x1,y1,x2,y2]); raw_ball_conf.append(conf); raw_ball_class_id.append(32)
    candidate_sv_detections = sv.Detections.empty()
    if raw_ball_xyxy: candidate_sv_detections = sv.Detections(xyxy=np.array(raw_ball_xyxy),confidence=np.array(raw_ball_conf),class_id=np.array(raw_ball_class_id))
    tracked_ball_raw_sv_detections = ball_tracker.update(candidate_sv_detections)
    if len(tracked_ball_raw_sv_detections) > 0:
        raw_ball_data = tracked_ball_raw_sv_detections[0]
        b_raw_box = tuple(map(int,raw_ball_data.xyxy[0])); b_raw_conf = float(raw_ball_data.confidence[0]) if raw_ball_data.confidence is not None and len(raw_ball_data.confidence)>0 else 0.5
        b_raw_center = get_center(b_raw_box)
        ball_kalman_filter.correct(b_raw_center); b_smoothed_center = ball_kalman_filter.predict()
        ball_result_for_storage = BallInfoBase(box=b_raw_box, center=b_smoothed_center, conf=b_raw_conf)
        if ball_motion_analyzer: ball_motion_analyzer.update(b_smoothed_center); ball_moving_status = ball_motion_analyzer.is_moving()
        if return_annotated_image and ball_annotator:
            annotated_frame = ball_annotator.annotate(annotated_frame, tracked_ball_raw_sv_detections)
            cv2.circle(annotated_frame, b_smoothed_center, 5, (0,0,255), -1)
            if ball_moving_status is not None: cv2.putText(annotated_frame, "Ball Moving" if ball_moving_status else "Ball Stationary", (frame_width-220,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    # if debug_print: print(f"  Ball detected: {'Yes' if ball_result_for_storage else 'No'}")

    pose_results = pose_model(frame, verbose=False)[0]; all_raw_poses_in_frame = []
    if pose_results.keypoints is not None and pose_results.boxes is not None:
        keypoints_data=pose_results.keypoints.xy; conf_data=pose_results.keypoints.conf; boxes_data=pose_results.boxes.xyxy
        num_poses_detected = keypoints_data.shape[0]
        if num_poses_detected == boxes_data.shape[0] and \
           (conf_data is None or num_poses_detected == conf_data.shape[0]):
            for i in range(num_poses_detected):
                current_conf_tensor = conf_data[i] if conf_data is not None else np.zeros(keypoints_data.shape[1])
                raw_kpts_dict=get_keypoints_dict(keypoints_data[i], current_conf_tensor)
                pose_bbox=tuple(map(int,boxes_data[i]))
                if raw_kpts_dict and is_inside_polygon(get_center(pose_bbox),PLAYING_FIELD_POLYGON):
                     all_raw_poses_in_frame.append({'keypoints':raw_kpts_dict,'box':pose_bbox})
    # if debug_print: print(f"  Raw poses detected in frame (after polygon filter): {len(all_raw_poses_in_frame)}")

    players_with_pose_for_storage: List[PlayerPoseInfoBase] = []
    MIN_IOU_FOR_POSE_ASSOCIATION = 0.1

    for p_info_consolidated in consolidated_players_this_frame:
        player_id_str=p_info_consolidated['player_id']; player_consolidated_box=p_info_consolidated['box_display']
        best_match_raw_pose_kpts_dict=None; best_match_iou=0.0
        # if debug_print: print(f"    Trying to associate pose for player {player_id_str} with box {player_consolidated_box}")
        for i, raw_pose_info in enumerate(all_raw_poses_in_frame):
            current_raw_kpts=raw_pose_info['keypoints']; current_pose_box=raw_pose_info['box']
            iou=calculate_iou(player_consolidated_box,current_pose_box)
            # if debug_print:
                 # print(f"      Player {player_id_str} vs Raw Pose {i} (box {current_pose_box}): IoU = {iou:.4f}")
                 # if return_annotated_image: # Draw pose boxes for debug
                    # cv2.rectangle(annotated_frame, (current_pose_box[0], current_pose_box[1]),
                                  # (current_pose_box[2], current_pose_box[3]), (0, 255, 0), 1)
                    # cv2.putText(annotated_frame, f"P{i}", (current_pose_box[0], current_pose_box[1]-5),
                                # cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            if iou > best_match_iou and iou > MIN_IOU_FOR_POSE_ASSOCIATION:
                # if debug_print: print(f"        Found potential match: Player {player_id_str} with Raw Pose {i}, IoU = {iou:.4f}")
                best_match_iou=iou
                best_match_raw_pose_kpts_dict=current_raw_kpts
        if best_match_raw_pose_kpts_dict and player_pose_smoother:
            # if debug_print: print(f"      SUCCESS: Associated Player {player_id_str} with a pose (IoU: {best_match_iou:.4f}). Smoothing...")
            smoothed_kpts_dict=player_pose_smoother.smooth(player_id_str,best_match_raw_pose_kpts_dict)
            pydantic_kpts={name:KeypointData(**data) for name,data in smoothed_kpts_dict.items()}
            player_pose_base_item = PlayerPoseInfoBase(
                player_id=player_id_str, box_display=p_info_consolidated['box_display'],
                center_display=p_info_consolidated['center_display'], keypoints=pydantic_kpts,
                type=p_info_consolidated['type']
            )
            players_with_pose_for_storage.append(player_pose_base_item)
            if player_id_str in current_frame_player_tracking_info_for_drawing:
                 current_frame_player_tracking_info_for_drawing[player_id_str]['current_pose_smoothed'] = smoothed_kpts_dict
        # elif debug_print:
            # print(f"      FAILURE: No suitable pose found for Player {player_id_str} (best IoU was {best_match_iou:.4f})")
    if debug_print: print(f"  Players associated with a pose: {len(players_with_pose_for_storage)}")

    if return_annotated_image:
        for p_draw_info in current_frame_player_tracking_info_for_drawing.values(): # Use .values()
            smoothed_kpts_for_drawing = p_draw_info.get('current_pose_smoothed')
            if smoothed_kpts_for_drawing: # Check if smoothed keypoints exist
                p_color = p_draw_info['color']
                # Draw skeleton connections
                for i_skel,j_skel in SKELETON_CONNECTIONS:
                    kp1_name = KEYPOINT_NAMES[i_skel]
                    kp2_name = KEYPOINT_NAMES[j_skel]
                    kp1_info = smoothed_kpts_for_drawing.get(kp1_name)
                    kp2_info = smoothed_kpts_for_drawing.get(kp2_name)
                    if kp1_info and kp1_info.get('xy') and kp2_info and kp2_info.get('xy'): # Check both keypoints and their 'xy'
                        pt1 = kp1_info['xy']
                        pt2 = kp2_info['xy']
                        # Sanity check coordinates before drawing (optional, but can prevent errors with extreme values)
                        # if 0 <= pt1[0] < frame_width and 0 <= pt1[1] < frame_height and \
                        #    0 <= pt2[0] < frame_width and 0 <= pt2[1] < frame_height:
                        cv2.line(annotated_frame, pt1, pt2, p_color, 2)
                # Draw keypoints themselves
                for kp_data_draw in smoothed_kpts_for_drawing.values(): # Iterate through values of the dict
                    if kp_data_draw.get('xy'):
                        pt_draw = kp_data_draw['xy']
                        # if 0 <= pt_draw[0] < frame_width and 0 <= pt_draw[1] < frame_height:
                        cv2.circle(annotated_frame, pt_draw, 3, p_color, -1)

    possessor_final_id: Optional[str] = None
    if ball_result_for_storage and ball_result_for_storage.center:
        ball_center_smoothed=ball_result_for_storage.center; min_ball_dist_sq=float('inf'); POSSESSION_THRESHOLD_SQ=5000
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
    if debug_print: print(f"  Possessor ID: {possessor_final_id if possessor_final_id else 'None'}")

    annotated_image_b64_str = encode_image(annotated_frame) if return_annotated_image else None

    return FrameAnalysisResult(
        frame_number_processed=frame_number_processed, video_filename_processed=video_filename_processed,
        players=players_with_pose_for_storage, ball=ball_result_for_storage,
        possessor_id=possessor_final_id, annotated_image_base64=annotated_image_b64_str,
        num_consolidated_players=len(consolidated_players_this_frame),
        num_raw_poses_in_frame=len(all_raw_poses_in_frame),
        num_players_with_pose_and_id=len(players_with_pose_for_storage),
        ball_is_moving=ball_moving_status
    )

# --- FastAPI Endpoint ---
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
    if frame_number % 10 == 0: print(f"Req-{req_id_suffix}: Frame {frame_number}, Vid '{video_filename}'. Save: {save_to_json_dir is not None}. ReturnImg: {return_annotated_image}")

    contents = await image_file.read()
    frame_cv = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame_cv is None:
        print(f"Req-{req_id_suffix}: ERROR - Could not decode image for frame {frame_number}.")
        raise HTTPException(status_code=400, detail="Could not decode image.")

    polygon_coords_list: Optional[List[Tuple[int, int]]] = None
    if playing_field_polygon_coords_json:
        try:
            parsed_json = json.loads(playing_field_polygon_coords_json)
            if not (isinstance(parsed_json, list) and len(parsed_json) >= 3 and
                    all(isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(c, (int, float)) for c in pt) for pt in parsed_json)):
                raise ValueError("Invalid polygon format. Must be list of [x,y] pairs, min 3 points.")
            polygon_coords_list = [(int(pt[0]), int(pt[1])) for pt in parsed_json]
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Req-{req_id_suffix}: ERROR - Invalid polygon JSON for frame {frame_number}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid playing_field_polygon_coords_json: {e}")

    try:
        analysis_output: FrameAnalysisResult = process_frame_for_analysis(
            frame_cv, frame_number_processed=frame_number, video_filename_processed=video_filename,
            playing_field_polygon_coords=polygon_coords_list, return_annotated_image=return_annotated_image
        )
        analysis_output.performance_feedback = []
        if save_to_json_dir:
            try:
                sane_video_filename = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in video_filename).rstrip()
                if not sane_video_filename: sane_video_filename = "unknown_video"
                video_specific_dir = Path(save_to_json_dir) / sane_video_filename
                video_specific_dir.mkdir(parents=True, exist_ok=True)
                json_filename = f"frame_{frame_number:06d}.json"
                json_filepath = video_specific_dir / json_filename
                storage_data = FrameDataForStorage(
                    frame_number_processed=analysis_output.frame_number_processed,
                    video_filename_processed=analysis_output.video_filename_processed,
                    players=analysis_output.players, ball=analysis_output.ball,
                    possessor_id=analysis_output.possessor_id,
                    num_consolidated_players=analysis_output.num_consolidated_players,
                    num_raw_poses_in_frame=analysis_output.num_raw_poses_in_frame,
                    num_players_with_pose_and_id=analysis_output.num_players_with_pose_and_id,
                    ball_is_moving=analysis_output.ball_is_moving
                )
                dict_to_save = storage_data.model_dump()
                with open(json_filepath, 'w') as f: json.dump(dict_to_save, f, indent=2)
                if frame_number % 10 == 0: print(f"Req-{req_id_suffix}: ✓ Saved JSON for frame {frame_number}")
            except Exception as e_save:
                print(f"Req-{req_id_suffix}: !!! ERROR saving JSON for frame {frame_number}: {e_save} !!!"); traceback.print_exc()

        processing_time_end = time.time()
        if frame_number % 10 == 0: print(f"Req-{req_id_suffix}: Frame {frame_number} processed in {processing_time_end - request_time_start:.4f} secs.")
        return analysis_output
    except HTTPException as e:
        print(f"Req-{req_id_suffix}: HTTP Exception during processing frame {frame_number}: {e.detail}"); raise e
    except RuntimeError as e:
        print(f"Req-{req_id_suffix}: Runtime error during processing frame {frame_number}: {e}"); traceback.print_exc(); raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Req-{req_id_suffix}: !!! UNEXPECTED error processing frame {frame_number}: {e} !!!"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn server for v5_mvp.py (Kalman & Drawing Refinement)...")
    uvicorn.run("v5_1_mvp:app", host="0.0.0.0", port=8000, workers=1)