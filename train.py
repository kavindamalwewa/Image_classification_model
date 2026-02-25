import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================
# DATASET PATH
# =============================
dataset_path = "Dataset"

# =============================
# SETTINGS
# =============================
img_size = 160
batch_size = 32

# =============================
# DATA AUGMENTATION
# =============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes

# =============================
# CNN MODEL (FROM SCRATCH)
# =============================
model = models.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(img_size, img_size, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# =============================
# COMPILE
# =============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =============================
# CALLBACK (ONLY LR REDUCTION)
# =============================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

# =============================
# TRAIN
# =============================
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[reduce_lr]
)

# =============================
# SAVE TRAINING HISTORY
# =============================
import json

with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("✅ Training history saved!")

# =============================
# EVALUATION
# =============================
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys()
)
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =============================
# SAVE MODEL
# =============================
model.save("vehicle_model_cnn.h5")
print("✅ Model Saved!")

# =============================
# TRAINING & VALIDATION GRAPHS
# =============================
epochs_ran = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(14,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs_ran, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_ran, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Training")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1,2,2)
plt.plot(epochs_ran, history.history['loss'], label='Training Loss')
plt.plot(epochs_ran, history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss Over Training")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================
# VALIDATION IMAGE PREDICTIONS
# =============================
class_labels = list(validation_generator.class_indices.keys())

images, labels = next(validation_generator)

pred_probs = model.predict(images)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = np.argmax(labels, axis=1)

plt.figure(figsize=(15,8))

for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(images[i])
    plt.axis('off')

    pred_label = class_labels[pred_classes[i]]
    true_label = class_labels[true_classes[i]]
    confidence = np.max(pred_probs[i]) * 100

    color = "green" if pred_classes[i] == true_classes[i] else "red"

    plt.title(
        f"P: {pred_label}\nT: {true_label}\n{confidence:.1f}%",
        color=color,
        fontsize=9
    )

plt.suptitle("Validation Image Predictions")
plt.tight_layout()
plt.show()

# =============================
# MISCLASSIFIED IMAGES ONLY
# =============================
plt.figure(figsize=(15,8))
count = 0

for i in range(len(images)):
    if pred_classes[i] != true_classes[i]:

        plt.subplot(3,4,count+1)
        plt.imshow(images[i])
        plt.axis('off')

        plt.title(
            f"Wrong!\nP:{class_labels[pred_classes[i]]}\nT:{class_labels[true_classes[i]]}",
            color="red",
            fontsize=9
        )

        count += 1
        if count == 12:
            break

plt.suptitle("Misclassified Validation Images")
plt.tight_layout()
plt.show()