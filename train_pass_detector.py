import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import os # For os.path.abspath

print("--- train_pass_detector.py starting ---") # Script start

# --- Configuration ---
DATA_ROOT = "./pass_data_for_training"
SEQUENCES_FILE = Path(DATA_ROOT) / "pass_action_sequences.npy"
LABELS_FILE = Path(DATA_ROOT) / "pass_action_labels.npy"
MODEL_SAVE_PATH = Path(DATA_ROOT) / "pass_detector_lstm.pth"

# Model Hyperparameters
INPUT_SIZE = None
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32
TEST_SPLIT_SIZE = 0.2
# --- End Configuration ---

print(f"CONFIG: DATA_ROOT='{os.path.abspath(DATA_ROOT)}'")
print(f"CONFIG: SEQUENCES_FILE='{SEQUENCES_FILE.resolve()}'")
print(f"CONFIG: LABELS_FILE='{LABELS_FILE.resolve()}'")

# 1. Load Data
print(f"\nSTEP 1: Loading data...")
if not SEQUENCES_FILE.exists():
    print(f"  CRITICAL ERROR: SEQUENCES_FILE '{SEQUENCES_FILE.resolve()}' does not exist.")
    exit()
else:
    print(f"  Found SEQUENCES_FILE: '{SEQUENCES_FILE.resolve()}'")

if not LABELS_FILE.exists():
    print(f"  CRITICAL ERROR: LABELS_FILE '{LABELS_FILE.resolve()}' does not exist.")
    exit()
else:
    print(f"  Found LABELS_FILE: '{LABELS_FILE.resolve()}'")

try:
    sequences = np.load(SEQUENCES_FILE)
    labels = np.load(LABELS_FILE) # This is 'labels' for the entire dataset
    print(f"  Successfully loaded .npy files.")
except Exception as e:
    print(f"  CRITICAL ERROR: Failed to load .npy files: {e}")
    exit()

if not hasattr(sequences, 'size') or sequences.size == 0:
    print(f"  CRITICAL ERROR: 'sequences' data is empty or not a valid NumPy array. Size: {getattr(sequences, 'size', 'N/A')}")
    exit()
if not hasattr(labels, 'size') or labels.size == 0:
    print(f"  CRITICAL ERROR: 'labels' data is empty or not a valid NumPy array. Size: {getattr(labels, 'size', 'N/A')}")
    exit()

print(f"  Raw sequences shape: {sequences.shape}")
print(f"  Raw labels shape: {labels.shape}")
unique_labels_all, counts_all = np.unique(labels, return_counts=True) # Use different var name
print(f"  Unique labels and counts in full dataset: {dict(zip(unique_labels_all, counts_all))}")

if len(sequences.shape) < 3:
    print(f"  CRITICAL ERROR: 'sequences' array has shape {sequences.shape}, expected 3 dimensions (num_sequences, sequence_length, num_features).")
    exit()
INPUT_SIZE = sequences.shape[2]
print(f"  Determined INPUT_SIZE (num_features): {INPUT_SIZE}")

if len(sequences) != len(labels):
    print(f"  CRITICAL ERROR: Mismatch between number of sequences ({len(sequences)}) and labels ({len(labels)}).")
    exit()

if len(sequences) < 2 :
    print(f"  CRITICAL ERROR: Not enough samples ({len(sequences)}) to perform a train/test split. Need at least 2.")
    exit()

# Check if TEST_SPLIT_SIZE would result in empty train or test sets
# Simplified check: ensure at least 1 sample in both train and test after split.
# This calculation might need adjustment if stratify causes issues with very small minority classes.
min_samples_per_class_for_stratify = 2 # A common minimum for stratification to work well
if not all(c >= min_samples_per_class_for_stratify for c in counts_all):
     print(f"  WARNING: Some classes have fewer than {min_samples_per_class_for_stratify} samples. Stratification might be problematic or lead to empty splits.")

if int(len(sequences) * (1-TEST_SPLIT_SIZE)) < 1 or int(len(sequences) * TEST_SPLIT_SIZE) < 1:
    print(f"  CRITICAL ERROR: Test split size {TEST_SPLIT_SIZE} results in < 1 sample for train or test set with {len(sequences)} total samples.")
    exit()


# 2. Prepare Data for PyTorch
print(f"\nSTEP 2: Preparing data for PyTorch...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=labels
    )
    print(f"  Data split: Train size={len(X_train)}, Test size={len(X_test)}")
    unique_labels_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"    Unique labels and counts in y_train: {dict(zip(unique_labels_train, counts_train))}")

except ValueError as e_split:
    print(f"  CRITICAL ERROR during train_test_split: {e_split}")
    print(f"    This can happen if there are too few samples in a class for stratification with test_size={TEST_SPLIT_SIZE}.")
    print(f"    Consider reducing TEST_SPLIT_SIZE, getting more data for minority classes, or not stratifying if classes are extremely small.")
    exit()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
print(f"  Converted data to PyTorch tensors.")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

if len(train_dataset) == 0:
    print(f"  CRITICAL ERROR: Training dataset is empty after split and tensor conversion.")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"  Created DataLoaders. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

if len(train_loader) == 0 and len(train_dataset) > 0 : # Check if train_loader is empty but dataset is not
    print(f"  CRITICAL ERROR: Train DataLoader has 0 batches, but training dataset has {len(train_dataset)} samples. Check BATCH_SIZE ({BATCH_SIZE}).")
    exit()
elif len(train_loader) == 0 and len(train_dataset) == 0:
    print(f"  CRITICAL ERROR: Train DataLoader has 0 batches because training dataset is empty.")
    exit()


print(f"  Training samples: {len(X_train_tensor)}, Test samples: {len(X_test_tensor)}")
print(f"  Input feature size: {INPUT_SIZE}")

# 3. Define LSTM Model
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

print(f"\nSTEP 3: Defining model and device...")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"  Using device: {device}")

model = PassDetectorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
print("\nModel Architecture:")
print(model)

# 4. Define Loss Function and Optimizer
print(f"\nSTEP 4: Defining loss and optimizer...")

# --- Calculate weights for BCEWithLogitsLoss ---
# IMPORTANT: Use y_train (from the training split) to calculate weights, not the full 'labels' array
# This prevents data leakage from the test set into the training process.
num_neg_samples_train = np.sum(y_train == 0)
num_pos_samples_train = np.sum(y_train == 1)
print(f"  Training set composition: Negatives={num_neg_samples_train}, Positives={num_pos_samples_train}")

if num_pos_samples_train > 0 and num_neg_samples_train > 0 : # Ensure both classes are present in training
    pos_weight_value = num_neg_samples_train / num_pos_samples_train
    print(f"  Calculated pos_weight for BCEWithLogitsLoss (based on training set): {pos_weight_value:.2f}")
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
elif num_pos_samples_train == 0:
    print("  WARNING: No positive samples (label 1) found in the training set. Using standard BCEWithLogitsLoss. Model will likely only predict negative.")
    criterion = nn.BCEWithLogitsLoss()
elif num_neg_samples_train == 0:
    print("  WARNING: No negative samples (label 0) found in the training set. Using standard BCEWithLogitsLoss. Model will likely only predict positive.")
    criterion = nn.BCEWithLogitsLoss()
else: # Should not happen if previous checks for empty dataset passed
    print("  WARNING: Could not determine positive/negative sample counts properly. Using standard BCEWithLogitsLoss.")
    criterion = nn.BCEWithLogitsLoss()
# --- End weight calculation ---

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"  Optimizer: Adam (lr={LEARNING_RATE})")


# 5. Training Loop
print(f"\nSTEP 5: Starting training for {NUM_EPOCHS} epochs...")
if len(train_loader) == 0:
    print("  CRITICAL ERROR: Train DataLoader is empty, cannot start training loop. Check BATCH_SIZE and number of training samples.")
    exit()

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for i, (batch_sequences, batch_labels) in enumerate(train_loader):
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_sequences)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if (i + 1) % max(1, (len(train_loader) // 2)) == 0 and len(train_loader) > 1 :
             print(f'    Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}')

    avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
    print(f'  Epoch [{epoch+1}/{NUM_EPOCHS}], Average Training Loss: {avg_epoch_loss:.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_preds_val = []
        all_true_val = []
        if len(test_loader) > 0: # Ensure test_loader is not empty
            for batch_sequences, batch_labels in test_loader:
                batch_sequences = batch_sequences.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_sequences)
                predicted_probs = torch.sigmoid(outputs)
                predicted_labels = (predicted_probs > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted_labels == batch_labels).sum().item()
                all_preds_val.extend(predicted_labels.cpu().numpy().flatten())
                all_true_val.extend(batch_labels.cpu().numpy().flatten())

            accuracy = 100 * correct / total if total > 0 else 0
            print(f'    Validation Accuracy on {total} test samples: {accuracy:.2f} %')
            
            if total > 0 and (epoch % 5 == 0 or epoch == NUM_EPOCHS - 1):
                print("    Validation Classification Report:")
                if all_true_val and all_preds_val:
                     print(classification_report(all_true_val, all_preds_val, target_names=['Not Pass', 'Pass'], zero_division=0))
                else:
                    print("      Not enough validation data/predictions for classification report this epoch.")
        else:
            print("    WARNING: Test DataLoader is empty. Skipping validation for this epoch.")


print("\nSTEP 6: Training finished.")

# 6. Save the Model
print(f"\nSTEP 7: Saving model to {MODEL_SAVE_PATH.resolve()}...")
try:
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Model saved successfully to {MODEL_SAVE_PATH.resolve()}")
except Exception as e_save:
    print(f"  CRITICAL ERROR: Failed to save model: {e_save}")


# 7. Final Evaluation
print("\n--- Final Evaluation on Test Set ---")
model.eval()
y_pred_list = []
y_true_list = []
if len(test_loader) > 0:
    with torch.no_grad():
        for batch_sequences, batch_labels in test_loader:
            batch_sequences = batch_sequences.to(device)
            outputs = model(batch_sequences)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).float()
            y_pred_list.extend(predicted_labels.cpu().numpy().flatten().tolist())
            y_true_list.extend(batch_labels.cpu().numpy().flatten().tolist())

    if y_true_list and y_pred_list:
        final_accuracy = accuracy_score(y_true_list, y_pred_list)
        print(f"Accuracy: {final_accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true_list, y_pred_list))
        print("\nClassification Report:")
        print(classification_report(y_true_list, y_pred_list, target_names=['Not Pass', 'Pass'], zero_division=0))
    else:
        print("  No data in test set or no predictions made for final evaluation.")
else:
    print("  Test DataLoader is empty, skipping final evaluation.")

print("\n--- train_pass_detector.py finished ---")