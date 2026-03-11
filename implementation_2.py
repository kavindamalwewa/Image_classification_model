import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# =============================
# DOWNLOADS
# Dataset : https://drive.google.com/file/d/1aqCxrwoU7-w5CO1k9mYcU2uLFP0TuyUy/view?usp=sharing
# Model   : https://drive.google.com/file/d/1ovi_R_7v4_eLFCxvnkcxRyGYcXeAbeQl/view?usp=drive_link
# =============================

# DATASET PATH

dataset_path = "Dataset"

# SETTINGS

img_size = 160
batch_size = 32

# DATA AUGMENTATION

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


# SAVE LABELS

labels = list(train_generator.class_indices.keys())

with open("labels.json", "w") as f:
    json.dump(labels, f)

print("Labels saved")


# LOAD PRETRAINED MODEL

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size,img_size,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# BUILD MODEL

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(256,activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes,activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

model.summary()

# COMPILE

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# CALLBACKS

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_vehicle_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# TRAIN

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop, checkpoint]
)

# SAVE TRAINING HISTORY

with open("training_history.json","w") as f:
    json.dump(history.history,f)

print("Training history saved")

# CONFUSION MATRIX

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions,axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys()
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# CLASSIFICATION REPORT

print("\nClassification Report:\n")
print(classification_report(y_true,y_pred))

# SAVE MODEL

model.save("model_v2.h5")

print("Model Saved")

# TRAINING CURVES

epochs_ran = range(1,len(history.history['accuracy'])+1)

plt.figure(figsize=(14,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs_ran,history.history['accuracy'],label="Training Accuracy")
plt.plot(epochs_ran,history.history['val_accuracy'],label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1,2,2)
plt.plot(epochs_ran,history.history['loss'],label="Training Loss")
plt.plot(epochs_ran,history.history['val_loss'],label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# VALIDATION IMAGE PREDICTIONS

class_labels = list(validation_generator.class_indices.keys())

images, labels = next(validation_generator)

pred_probs = model.predict(images)
pred_classes = np.argmax(pred_probs,axis=1)
true_classes = np.argmax(labels,axis=1)

plt.figure(figsize=(15,8))

for i in range(12):

    plt.subplot(3,4,i+1)
    plt.imshow(images[i])
    plt.axis('off')

    pred_label = class_labels[pred_classes[i]]
    true_label = class_labels[true_classes[i]]

    confidence = np.max(pred_probs[i])*100

    color = "green" if pred_classes[i]==true_classes[i] else "red"

    plt.title(
        f"P:{pred_label}\nT:{true_label}\n{confidence:.1f}%",
        color=color,
        fontsize=9
    )

plt.suptitle("Validation Predictions")
plt.tight_layout()
plt.show()

# MISCLASSIFIED IMAGES

plt.figure(figsize=(15,8))
count=0

for i in range(len(images)):

    if pred_classes[i]!=true_classes[i]:

        plt.subplot(3,4,count+1)
        plt.imshow(images[i])
        plt.axis('off')

        plt.title(
            f"Wrong\nP:{class_labels[pred_classes[i]]}\nT:{class_labels[true_classes[i]]}",
            color="red",
            fontsize=9
        )

        count+=1

        if count==12:
            break

plt.suptitle("Misclassified Images")
plt.tight_layout()
plt.show()