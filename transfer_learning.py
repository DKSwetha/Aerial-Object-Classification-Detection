import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#  IMPORT GENERATORS
from data_preprocessing import (
    train_generator,
    val_generator,
    test_generator,
    IMG_SIZE
)

IMG_HEIGHT, IMG_WIDTH = IMG_SIZE
EPOCHS = 30


#  1. FUNCTION: BUILD TRANSFER LEARNING MODEL
def build_transfer_model(base_model_fn, model_name):

    # Load base model — no top (we add our own head)
    base = base_model_fn(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base.trainable = False   # Freeze all base layers initially

    # Build model
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)   # Binary output

    model = Model(inputs=base.input, outputs=output, name=model_name)
    return model, base


#  2. FUNCTION: TRAIN A MODEL

def train_model(model, model_name):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall")
        ]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"best_{model_name}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\n Training {model_name} (Phase 1 — frozen base)...\n")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    return history


#  3. FUNCTION: FINE-TUNE (unfreeze top layers)

def fine_tune(model, base, model_name, unfreeze_from=-20):
    base.trainable = True

    # Freeze all except last `unfreeze_from` layers
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall")
        ]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=f"best_{model_name}_finetuned.keras",
                        monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    print(f"\n Fine-tuning {model_name} (Phase 2 — last 20 layers unfrozen)...\n")
    history_ft = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        callbacks=callbacks
    )
    return history_ft


#  4. FUNCTION: EVALUATE MODEL
def evaluate_model(model, model_name):
    print(f"\n Evaluating {model_name} on Test Set...")
    test_generator.reset()

    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_generator, verbose=0)
    f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-8)

    print(f"\n  Test Accuracy  : {test_acc:.4f}")
    print(f"  Test Precision : {test_prec:.4f}")
    print(f"  Test Recall    : {test_rec:.4f}")
    print(f"  Test F1-Score  : {f1:.4f}")

    # Confusion Matrix
    test_generator.reset()
    y_pred = (model.predict(test_generator) > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} — Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=150)
    plt.show()
    print(f"Confusion matrix saved as '{model_name}_confusion_matrix.png'")

    print(f"\n Classification Report ({model_name}):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "model":     model_name,
        "accuracy":  round(test_acc,  4),
        "precision": round(test_prec, 4),
        "recall":    round(test_rec,  4),
        "f1_score":  round(f1,        4)
    }

#  5. FUNCTION: PLOT TRAINING CURVES
def plot_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    fname = f"{model_name}_curves.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Curves saved as '{fname}'")


#  6. TRAIN ALL THREE MODELS

all_metrics = []

model_configs = [
    (ResNet50,       "ResNet50"),
    (MobileNetV2,    "MobileNetV2"),
    (EfficientNetB0, "EfficientNetB0"),
]

for base_fn, name in model_configs:
    print(f"\n{'='*55}")
    print(f"  MODEL: {name}")
    print(f"{'='*55}")

    model, base = build_transfer_model(base_fn, name)
    model.summary()

    # Phase 1: Train with frozen base
    history = train_model(model, name)
    plot_history(history, name)

    # Phase 2: Fine-tune last 20 layers
    history_ft = fine_tune(model, base, name)
    plot_history(history_ft, f"{name}_finetuned")

    # Evaluate
    metrics = evaluate_model(model, name)
    all_metrics.append(metrics)


#  7. SAVE ALL METRICS FOR COMPARISON
with open("transfer_learning_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n All transfer learning metrics saved to 'transfer_learning_metrics.json'")
