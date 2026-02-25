# =============================
# TRAINING HISTORY GRAPHS
# =============================
import matplotlib.pyplot as plt

epochs_ran = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(14,5))

# -----------------------------
# Accuracy Graph
# -----------------------------
plt.subplot(1,2,1)
plt.plot(epochs_ran, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_ran, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Training")
plt.legend()
plt.grid(True)

# -----------------------------
# Loss Graph
# -----------------------------
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