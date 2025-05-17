import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import uuid as uuid_lib

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
    (5, 11), (6, 12), (5, 6), (11, 12), (5, 7), (7, 9),
    (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 0), (6, 0), (0, 1), (0, 2), (1, 3), (2, 4)
]

PLAYING_FIELD_POLYGON = None

# --- Helper Functions ---
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_distance_sq(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def is_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_inside_polygon(point, polygon):
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

def get_keypoints_dict(keypoints_tensor, conf_tensor):
    kpts = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(keypoints_tensor) and i < len(conf_tensor):
            x, y = keypoints_tensor[i]
            conf = conf_tensor[i]
            if conf > 0.3:
                kpts[name] = {'xy': (int(x), int(y)), 'conf': conf.item()}
    return kpts

def assign_static_uuid(center, uuid_map):
    for prev_center, uid in uuid_map.items():
        if get_distance_sq(center, prev_center) < 1000:
            return uid
    new_uuid = str(uuid_lib.uuid4())
    uuid_map[center] = new_uuid
    return new_uuid

def detect_aruco_markers(frame, id_to_uuid):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    uuid_labels = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_uuid = id_to_uuid.get(marker_id, str(uuid_lib.uuid4()))
            id_to_uuid[marker_id] = marker_uuid
            corner = corners[i][0][0].astype(int)
            cv2.putText(frame, f"UUID: {marker_uuid[:8]}", tuple(corner),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            uuid_labels[marker_id] = marker_uuid
    return uuid_labels

# --- Load Models ---
print("Loading models...")
object_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")
print("Models loaded.")

# --- Webcam Setup ---
CAMERA_INDEX = 2
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if PLAYING_FIELD_POLYGON is None:
    PLAYING_FIELD_POLYGON = np.array([
        [0, 0], [frame_width - 1, 0],
        [frame_width - 1, frame_height - 1], [0, frame_height - 1]
    ], np.int32)

player_uuid_map = {}
id_to_uuid_map = {}

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_aruco_markers(frame, id_to_uuid_map)

    object_detections_result = object_model(frame, verbose=False)[0]
    detected_players_boxes = []
    for det in object_detections_result.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        center = get_center((x1, y1, x2, y2))
        label = object_model.names[int(det.cls)]
        if label == 'person' and is_inside_polygon(center, PLAYING_FIELD_POLYGON):
            detected_players_boxes.append({'box': (x1, y1, x2, y2), 'center': center})

    pose_results = pose_model(frame, verbose=False)[0]
    all_poses = []
    if pose_results.keypoints and pose_results.keypoints.xy is not None:
        for i in range(len(pose_results.keypoints.xy)):
            kpts_tensor = pose_results.keypoints.xy[i]
            conf_tensor = pose_results.keypoints.conf[i]
            keypoints_dict = get_keypoints_dict(kpts_tensor, conf_tensor)
            if keypoints_dict:
                core_kpt = keypoints_dict.get("nose") or keypoints_dict.get("left_hip")
                if core_kpt and is_inside_polygon(core_kpt['xy'], PLAYING_FIELD_POLYGON):
                    all_poses.append(keypoints_dict)

    players_with_pose = []
    unassigned_poses = list(all_poses)
    for box_info in detected_players_boxes:
        box = box_info['box']
        center = box_info['center']
        matched_pose = None
        for pose in unassigned_poses:
            for ref in ["nose", "left_hip", "right_hip"]:
                if ref in pose and is_inside_box(pose[ref]['xy'], box):
                    matched_pose = pose
                    break
            if matched_pose:
                break
        if matched_pose:
            player_uuid = assign_static_uuid(center, player_uuid_map)
            players_with_pose.append({'box': box, 'center': center, 'keypoints': matched_pose, 'uuid': player_uuid})
            unassigned_poses.remove(matched_pose)

    for player in players_with_pose:
        box = player['box']
        uuid_label = player['uuid'][:8]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, f"UUID: {uuid_label}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for i, j in SKELETON_CONNECTIONS:
            kp1, kp2 = KEYPOINT_NAMES[i], KEYPOINT_NAMES[j]
            if kp1 in player['keypoints'] and kp2 in player['keypoints']:
                pt1 = player['keypoints'][kp1]['xy']
                pt2 = player['keypoints'][kp2]['xy']
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)

    cv2.imshow("YOLO Pose Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
