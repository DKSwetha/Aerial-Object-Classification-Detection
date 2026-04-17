import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


from data_preprocessing import (
    train_generator,
    val_generator,
    test_generator,
    IMG_SIZE
)

IMG_HEIGHT, IMG_WIDTH = IMG_SIZE
EPOCHS = 30   # EarlyStopping will stop before this if needed


#  1. BUILD CUSTOM CNN MODEL
def build_custom_cnn():
    model = Sequential([

        # ── Block 1 ──────────────────────────
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # ── Block 2 ──────────────────────────
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # ── Block 3 ──────────────────────────
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # ── Classifier Head ──────────────────
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation="sigmoid")   # Binary output: Bird=0, Drone=1
    ])
    return model

cnn_model = build_custom_cnn()


#  2. COMPILE MODEL
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall")
    ]
)

cnn_model.summary()


#  3. CALLBACKS
callbacks = [
    # Stop training if val_loss doesn't improve for 5 epochs
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Save the best model automatically
    ModelCheckpoint(
        filepath="best_custom_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]


#  4. TRAIN MODEL
print("\n Starting Custom CNN Training...\n")

history = cnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

print("\n Training complete! Best model saved as 'best_custom_cnn.keras'")


#  5. PLOT TRAINING CURVES
def plot_history(history, model_name="Custom CNN"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_curves.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Training curves saved as '{filename}'")

plot_history(history, "Custom CNN")


#  6. EVALUATE ON TEST SET
print("\n Evaluating on Test Set...")
test_loss, test_acc, test_prec, test_rec = cnn_model.evaluate(test_generator)

# F1-Score (manual calculation)
f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-8)

print(f"\n  Test Accuracy  : {test_acc:.4f}")
print(f"  Test Precision : {test_prec:.4f}")
print(f"  Test Recall    : {test_rec:.4f}")
print(f"  Test F1-Score  : {f1:.4f}")


#  7. CONFUSION MATRIX & CLASSIFICATION REPORT
# Reset generator before predicting
test_generator.reset()
y_pred_probs = cnn_model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Custom CNN — Confusion Matrix", fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("custom_cnn_confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved as 'custom_cnn_confusion_matrix.png'")

# Classification Report
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


#  8. SAVE METRICS FOR COMPARISON LATER
cnn_metrics = {
    "model": "Custom CNN",
    "accuracy":  round(test_acc,  4),
    "precision": round(test_prec, 4),
    "recall":    round(test_rec,  4),
    "f1_score":  round(f1,        4)
}

import json
with open("cnn_metrics.json", "w") as f:
    json.dump(cnn_metrics, f, indent=2)

print("\n Metrics saved to 'cnn_metrics.json' ")
