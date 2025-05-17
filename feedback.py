import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle at point b (in degrees) formed by three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def Get_Perf_From_Pose(player_data, ball_center):
    feedback = []
    for player in player_data:
        pid = player.player_id
        keypoints = {k: {"xy": v.xy, "conf": v.conf} for k, v in player.keypoints.items()}
        pose_feedback = {"player_id": pid, "feedback": []}

        min_conf = 0.7

        # Posture analysis
        try:
            lsa, lha, lan = keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_ankle"]
            if all(kp["conf"] > min_conf for kp in [lsa, lha, lan]):
                left_posture = calculate_angle(lsa["xy"], lha["xy"], lan["xy"])
                if left_posture < 150:
                    pose_feedback["feedback"].append("You had a low stance—good posture for balance and agility.")
                else:
                    pose_feedback["feedback"].append("Your posture was upright—try bending knees more for better balance.")
        except KeyError:
            pass

        # Effort estimation
        try:
            y1, y2 = player.box_display[1], player.box_display[3]
            height = y2 - y1
            if height > 600:
                pose_feedback["feedback"].append("High effort detected—bounding box height suggests sprinting.")
            elif height < 400:
                pose_feedback["feedback"].append("You appeared stationary or upright for much of the frame.")
        except:
            pass

        # Ball proximity
        try:
            distance_to_ball = euclidean_distance(player.center_display, ball_center)
            if distance_to_ball < 200:
                pose_feedback["feedback"].append("You were close to the ball—great positioning.")
            elif distance_to_ball > 500:
                pose_feedback["feedback"].append("Try staying closer to the ball during active play.")
        except:
            pass

        feedback.append(pose_feedback)

    return feedback
