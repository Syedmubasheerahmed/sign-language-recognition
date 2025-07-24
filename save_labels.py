import os
import pickle

# Your dataset path
data_dir = r'D:\sign lang\dataset\asl_alphabet_train\asl_alphabet_train'

# This is the exact label order you want
desired_label_order = [
    'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Check which labels actually exist in your dataset folder and keep only those present
existing_labels = [label for label in desired_label_order if os.path.isdir(os.path.join(data_dir, label))]

# Print info as you process
for label in existing_labels:
    print(f"[INFO] Processing label: {label}")

# Save this exact label order to a file
with open('labels.pkl', 'wb') as f:
    pickle.dump(existing_labels, f)

print("[INFO] Labels saved to labels.pkl")
