import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = r'D:\sign lang\dataset\asl_alphabet_train\asl_alphabet_train'

desired_label_order = [
    'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

existing_labels = [label for label in desired_label_order if os.path.isdir(os.path.join(data_dir, label))]

with open('labels.pkl', 'wb') as f:
    pickle.dump(existing_labels, f)

print(f"Total labels: {len(existing_labels)}")

img_size = 64
data = []
labels = []

for label_idx, label in enumerate(existing_labels):
    label_path = os.path.join(data_dir, label)
    for img_name in os.listdir(label_path)[:3000]:
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype('float32') / 255.0
        data.append(img)
        labels.append(label_idx)
    print(f"Processed: {label} ({len(os.listdir(label_path)[:3000])} images)")

data = np.array(data)
labels = np.array(labels)

print(f"Dataset shape: {data.shape}, Labels shape: {labels.shape}")

labels_categorical = to_categorical(labels, num_classes=len(existing_labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels_categorical, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(existing_labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save('asl_sign_model.h5')
print("Model saved as asl_sign_model.h5")