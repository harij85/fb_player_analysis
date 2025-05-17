# pose_ball_tracking_kalman.py
import numpy as np
from collections import defaultdict, deque # Added deque here
import cv2
from typing import Optional, Tuple, Dict, DefaultDict, List

class KalmanFilter2D:
    def __init__(self, initial_coord: Optional[Tuple[float, float]] = None):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        # Consider a slightly higher measurementNoiseCov if keypoints are very jittery
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0 # Defaulted to 1.0 from earlier adjustment
        self.initialized = False

        if initial_coord is not None:
            init_x = float(initial_coord[0])
            init_y = float(initial_coord[1])
            self.kf.statePost = np.array([init_x, init_y, 0., 0.], dtype=np.float32)
            self.kf.statePre = np.array([init_x, init_y, 0., 0.], dtype=np.float32)
            self.initialized = True


    def correct(self, coord: Tuple[int, int]):
        measurement = np.array([[np.float32(coord[0])], [np.float32(coord[1])]])
        if not self.initialized:
            self.kf.statePost = np.array([float(coord[0]), float(coord[1]), 0., 0.], dtype=np.float32)
            self.kf.statePre = np.array([float(coord[0]), float(coord[1]), 0., 0.], dtype=np.float32)
            # Crucially, after setting statePre and statePost, a predict() call is needed before the first correct()
            # to properly initialize the filter's internal error covariance matrices if they weren't set manually.
            # However, since statePost IS the corrected state after this first measurement,
            # we can consider it initialized and ready for the next predict-correct cycle.
            self.initialized = True
            # A common pattern for first measurement:
            # 1. Set statePost (and statePre to same if no prior).
            # 2. The next cycle will be: predict() -> new_measurement -> correct().
            # For simplicity here, we'll just ensure it's initialized.
            # The very first kf.predict() in PlayerPoseSmoother will use this state.
            # Let's ensure correct() itself sets up the KF fully after first measurement.
            self.kf.predict() # This will use the just-set statePre.
            self.kf.correct(measurement) # This is the actual first correction.
        else:
            self.kf.correct(measurement)

    def predict(self) -> Optional[Tuple[int, int]]:
        if not self.initialized:
            return None
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])

class PlayerPoseSmoother:
    def __init__(self):
        self.trackers: DefaultDict[str, DefaultDict[str, KalmanFilter2D]] = defaultdict(lambda: defaultdict(lambda: KalmanFilter2D()))

    def smooth(self, player_id: str, keypoints_dict: Dict[str, Dict[str, any]]) -> Dict[str, Dict[str, any]]:
        smoothed_keypoints: Dict[str, Dict[str, any]] = {}
        
        for kp_name, kp_info in keypoints_dict.items():
            if 'xy' not in kp_info or kp_info['xy'] is None:
                continue

            current_xy: Tuple[int, int] = kp_info['xy']
            current_conf: float = kp_info['conf']

            kf = self.trackers[player_id][kp_name]

            kf.correct(current_xy) # This will initialize the KF on first call for this kp_name
            predicted_xy = kf.predict() # Then predict

            if predicted_xy is not None:
                smoothed_keypoints[kp_name] = {
                    'xy': predicted_xy,
                    'conf': current_conf 
                }
        return smoothed_keypoints


class BallMotionAnalyzer:
    def __init__(self, history: int = 5, threshold: float = 3.0):
        self.history: deque[Optional[Tuple[int, int]]] = deque(maxlen=history) # deque type hint
        self.threshold: float = threshold

    def update(self, center: Optional[Tuple[int, int]]):
        if center is None:
            return
        self.history.append(center)

    def is_moving(self) -> bool:
        if len(self.history) < 2:
            return False
        
        valid_history: List[Tuple[int,int]] = []
        for pt in self.history: # Iterate through the deque
            if isinstance(pt, tuple) and len(pt) == 2:
                 # Further check if elements are numbers (optional, but good for robustness)
                if isinstance(pt[0], (int, float)) and isinstance(pt[1], (int, float)):
                    valid_history.append(pt)

        if len(valid_history) < 2:
            return False

        deltas = [np.linalg.norm(np.array(valid_history[i]) - np.array(valid_history[i - 1]))
                  for i in range(1, len(valid_history))]
        
        if not deltas:
            return False
            
        avg_delta = np.mean(deltas)
        return avg_delta > self.threshold