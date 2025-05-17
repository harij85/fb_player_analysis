# process_pass_data_actual.py (Refactored with NaN handling for is_pass)
import json
import os
import numpy as np
import pandas as pd
# from collections import deque # Not used in this script directly
from typing import List, Dict, Any, Tuple, Optional
import shutil
from pathlib import Path

print("Script process_pass_data_actual.py starting...")

# --- Constants & Feature Engineering Helpers ---
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
def calculate_distance(p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> float:
    if p1 is None or p2 is None: return 1000.0
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_keypoint_coord(player_keypoints: Dict[str, Dict[str, Any]], kp_name: str) -> Optional[Tuple[float, float]]:
    kp_data = player_keypoints.get(kp_name)
    if kp_data and 'xy' in kp_data and isinstance(kp_data['xy'], (list, tuple)) and len(kp_data['xy']) == 2:
        try:
            return tuple(map(float, kp_data['xy']))
        except (ValueError, TypeError):
            return None
    return None

def extract_pass_features_from_frame(frame_data: Dict[str, Any], passer_id: Optional[str]) -> List[float]:
    features = []
    num_expected_features = 10
    passer_kps = {}
    player_center = None
    if passer_id:
        for p in frame_data.get("players", []):
            if p.get("player_id") == passer_id:
                passer_kps = p.get("keypoints", {})
                if p.get("center_display"):
                    try:
                        player_center = tuple(map(float, p.get("center_display")))
                        if len(player_center) != 2: player_center = None
                    except (ValueError, TypeError):
                        player_center = None
                break
    ball_center = None
    if frame_data.get("ball") and "center" in frame_data["ball"]:
        try:
            ball_center = tuple(map(float, frame_data["ball"]["center"]))
            if len(ball_center) != 2: ball_center = None
        except (ValueError, TypeError):
            ball_center = None
    rk_coord = get_keypoint_coord(passer_kps, "right_ankle")
    features.append(calculate_distance(rk_coord, ball_center))
    lk_coord = get_keypoint_coord(passer_kps, "left_ankle")
    features.append(calculate_distance(lk_coord, ball_center))
    features.append(calculate_distance(player_center, ball_center))
    if ball_center: features.extend(ball_center)
    else: features.extend([0.0, 0.0])
    if rk_coord: features.extend(rk_coord)
    else: features.extend([0.0, 0.0])
    if lk_coord: features.extend(lk_coord)
    else: features.extend([0.0, 0.0])
    features.append(1.0 if frame_data.get("ball_is_moving", False) else 0.0)
    if len(features) < num_expected_features:
        features.extend([1000.0] * (num_expected_features - len(features)))
    return features[:num_expected_features]
# --- End Helpers ---

def organize_and_extract_features(
    temp_json_root: str,
    final_data_root: str,
    annotations_csv: str,
    sequence_length: int,
    debug_print: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    if debug_print: print(f"  [organize_and_extract_features] Called with:\n    temp_json_root='{os.path.abspath(temp_json_root)}'\n    final_data_root='{os.path.abspath(final_data_root)}'\n    annotations_csv='{os.path.abspath(annotations_csv)}'")

    all_sequences: List[List[List[float]]] = []
    all_labels: List[int] = []

    Path(final_data_root).mkdir(parents=True, exist_ok=True)
    Path(final_data_root, "positive_pass_clips").mkdir(parents=True, exist_ok=True)
    Path(final_data_root, "negative_clips").mkdir(parents=True, exist_ok=True)

    try:
        annotations_df = pd.read_csv(annotations_csv)
        if debug_print: print(f"  [organize_and_extract_features] Loaded {len(annotations_df)} rows from '{annotations_csv}'")
    except FileNotFoundError:
        print(f"  [organize_and_extract_features] CRITICAL ERROR: Annotations CSV file not found at '{annotations_csv}'")
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    except pd.errors.EmptyDataError:
        print(f"  [organize_and_extract_features] CRITICAL ERROR: Annotations CSV file '{annotations_csv}' is empty.")
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    except Exception as e_csv:
        print(f"  [organize_and_extract_features] CRITICAL ERROR: Could not read CSV '{annotations_csv}': {e_csv}")
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)


    if annotations_df.empty: # This check is now a bit redundant due to try-except above, but harmless
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    for index, row in annotations_df.iterrows():
        original_clip_id = str(row["clip_folder_name"]).strip() # Add strip for safety

        is_pass_value = row["is_pass"]
        if pd.isna(is_pass_value):
            if debug_print: print(f"    WARNING: CSV row {index}: Missing 'is_pass' value for clip_id='{original_clip_id}'. Skipping this entry.")
            continue
        
        try:
            is_pass_label = int(float(is_pass_value)) # Convert to float first, then int, to handle "1.0"
            if is_pass_label not in [0, 1]:
                if debug_print: print(f"    WARNING: CSV row {index}: Invalid 'is_pass' value '{is_pass_value}' for clip_id='{original_clip_id}'. Expected 0 or 1. Skipping.")
                continue
        except ValueError:
            if debug_print: print(f"    WARNING: CSV row {index}: Cannot convert 'is_pass' value '{is_pass_value}' to int for clip_id='{original_clip_id}'. Skipping.")
            continue
            
        passer_id_val = row.get("passer_player_id") # Use .get for safety
        passer_id = str(passer_id_val).strip() if pd.notna(passer_id_val) and str(passer_id_val).strip() != "" else None
        
        if debug_print: print(f"    Processing CSV row {index}: clip_id='{original_clip_id}', is_pass={is_pass_label}, passer_id='{passer_id}'")

        source_clip_json_folder = Path(temp_json_root) / original_clip_id
        if debug_print: print(f"      Looking for source JSONs in: {source_clip_json_folder.resolve()}")
        if not source_clip_json_folder.is_dir():
            print(f"      WARNING: Source JSON folder '{source_clip_json_folder.resolve()}' NOT FOUND. Skipping this entry.")
            continue

        label_type_folder = "positive_pass_clips" if is_pass_label == 1 else "negative_clips"
        target_clip_organized_folder = Path(final_data_root) / label_type_folder / original_clip_id
        target_clip_organized_folder.mkdir(parents=True, exist_ok=True)
        if debug_print: print(f"      Target organized folder: {target_clip_organized_folder.resolve()}")

        json_files = sorted([f for f in os.listdir(source_clip_json_folder) if f.startswith("frame_") and f.endswith(".json")])
        if debug_print: print(f"      Found {len(json_files)} JSON files in source folder '{source_clip_json_folder}'.")
        
        num_copied = 0
        for fname in json_files:
            try:
                shutil.copy2(source_clip_json_folder / fname, target_clip_organized_folder / fname)
                num_copied+=1
            except Exception as e:
                print(f"      ERROR copying {fname} from {source_clip_json_folder} to {target_clip_organized_folder}: {e}")
        if debug_print: print(f"      Copied {num_copied} JSON files to target folder.")
        
        if len(json_files) < sequence_length:
            if debug_print: print(f"      Skipping feature extraction for '{original_clip_id}', not enough frames ({len(json_files)}) for sequence length {sequence_length}")
            continue
        
        clip_frame_features: List[List[float]] = []
        if debug_print: print(f"      Extracting features for {len(json_files)} frames...")
        for fname_idx, fname in enumerate(json_files):
            try:
                with open(target_clip_organized_folder / fname, 'r') as f:
                    frame_data = json.load(f)
                features = extract_pass_features_from_frame(frame_data, passer_id)
                if features:
                   clip_frame_features.append(features)
                elif debug_print:
                    print(f"          WARNING: No features extracted for frame {fname} (passer_id: {passer_id})")
            except json.JSONDecodeError:
                if debug_print: print(f"        Warning: Could not decode JSON {fname} in {target_clip_organized_folder}")
                continue
            except Exception as e_extract:
                if debug_print: print(f"        ERROR extracting features from {fname}: {e_extract}")
                continue
        
        if debug_print: print(f"      Extracted features for {len(clip_frame_features)} frames from clip '{original_clip_id}'.")
        if len(clip_frame_features) >= sequence_length:
            num_seq_in_clip = 0
            for i in range(len(clip_frame_features) - sequence_length + 1):
                sequence = clip_frame_features[i : i + sequence_length]
                all_sequences.append(sequence)
                all_labels.append(is_pass_label)
                num_seq_in_clip +=1
            if debug_print: print(f"      Generated {num_seq_in_clip} sequences for clip '{original_clip_id}'.")
        elif debug_print:
            print(f"      Not enough features extracted ({len(clip_frame_features)}) to form sequences for clip '{original_clip_id}'.")

    return np.array(all_sequences, dtype=np.float32), np.array(all_labels, dtype=np.int32)

if __name__ == "__main__":
    print("Inside __main__ block of process_pass_data_actual.py...")
    TEMP_JSON_ROOT = "./temp_json_output"
    FINAL_DATA_ROOT = "./pass_data_for_training"
    ANNOTATIONS_CSV = "./pass_annotations.csv" # MAKE SURE THIS FILE EXISTS AND IS CORRECT
    SEQUENCE_LENGTH = 15
    DEBUG_MODE = True

    print(f"CONFIG: TEMP_JSON_ROOT (abs): {os.path.abspath(TEMP_JSON_ROOT)}")
    print(f"CONFIG: FINAL_DATA_ROOT (abs): {os.path.abspath(FINAL_DATA_ROOT)}")
    print(f"CONFIG: ANNOTATIONS_CSV (abs): {os.path.abspath(ANNOTATIONS_CSV)}")
    print(f"CONFIG: SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    print(f"CONFIG: DEBUG_MODE: {DEBUG_MODE}")

    if not os.path.exists(ANNOTATIONS_CSV):
        print(f"CRITICAL ERROR: ANNOTATIONS_CSV file not found at '{os.path.abspath(ANNOTATIONS_CSV)}'.")
        print("Please create this CSV file with columns: clip_folder_name,is_pass,passer_player_id")
        exit()
    if not os.path.isdir(TEMP_JSON_ROOT):
        print(f"CRITICAL ERROR: TEMP_JSON_ROOT directory not found at '{os.path.abspath(TEMP_JSON_ROOT)}'.")
        exit()

    print("\nStarting data organization and feature extraction...")
    sequences, labels = organize_and_extract_features(
        TEMP_JSON_ROOT, FINAL_DATA_ROOT, ANNOTATIONS_CSV, SEQUENCE_LENGTH, debug_print=DEBUG_MODE
    )

    if sequences.size > 0:
        print(f"\n--- Processing Complete ---")
        print(f"  Data successfully organized into: {os.path.abspath(FINAL_DATA_ROOT)}")
        print(f"  Total sequences generated: {sequences.shape[0]}")
        print(f"  Shape of sequences array: {sequences.shape} (Sequences, Frames/Sequence, Features/Frame)")
        print(f"  Shape of labels array: {labels.shape}")
        print(f"  Number of positive samples (is_pass=1): {np.sum(labels == 1)}")
        print(f"  Number of negative samples (is_pass=0): {np.sum(labels == 0)}")
        sequences_save_path = Path(FINAL_DATA_ROOT) / "pass_action_sequences.npy"
        labels_save_path = Path(FINAL_DATA_ROOT) / "pass_action_labels.npy"
        np.save(sequences_save_path, sequences)
        np.save(labels_save_path, labels)
        print(f"  Saved processed sequences to: {sequences_save_path}")
        print(f"  Saved processed labels to: {labels_save_path}")
        print("\nDataset is ready for model training.")
    else:
        print("\n--- Processing Finished: No sequences were generated. ---")
        print("Please check the following:")
        print(f"1. Ensure '{os.path.abspath(ANNOTATIONS_CSV)}' exists, is not empty, and is formatted correctly (no missing 'is_pass' values).")
        print(f"2. Ensure '{os.path.abspath(TEMP_JSON_ROOT)}' exists and contains subfolders as listed in your CSV.")
        print(f"3. Ensure those subfolders (e.g., '{os.path.abspath(Path(TEMP_JSON_ROOT) / 'pass1')}') contain 'frame_XXXXXX.json' files.")
        print(f"4. Check that clip lengths are >= SEQUENCE_LENGTH ({SEQUENCE_LENGTH} frames).")
        print("5. Review any WARNING messages printed during processing if DEBUG_MODE was True.")