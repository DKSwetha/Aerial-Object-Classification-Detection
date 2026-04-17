import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#  CONFIGURATION — UPDATE THESE PATHS

DATASET_DIR  = r"D:\Labmentix\aerial object\object_detection_dataset\object_detection_Dataset"
PROJECT_DIR  = r"D:\Labmentix\aerial object\yolo_runs"
EPOCHS       = 50
IMG_SIZE     = 640     # YOLOv8 standard input size
BATCH_SIZE   = 16
MODEL_SIZE   = "yolov8n.pt"   # n=nano (fastest), s=small, m=medium

# Class names — must match your label class IDs
# class_id 0 = bird, class_id 1 = drone
CLASS_NAMES = ["bird", "drone"]


#  STEP 2 — INSPECT DATASET STRUCTURE

def inspect_yolo_dataset(dataset_dir):
    print("=" * 50)
    print("  YOLO DATASET INSPECTION")
    print("=" * 50)
    for split in ["train", "val", "test"]:
        img_path = os.path.join(dataset_dir, "images", split)
        lbl_path = os.path.join(dataset_dir, "labels", split)

        if not os.path.exists(img_path):
            print(f"  {split}: images folder not found at {img_path}")
            continue

        imgs = len([f for f in os.listdir(img_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        lbls = len([f for f in os.listdir(lbl_path)
                    if f.endswith(".txt")]) if os.path.exists(lbl_path) else 0

        print(f"\n  {split.upper()}")
        print(f"     Images : {imgs}")
        print(f"     Labels : {lbls}")
        if imgs != lbls:
            print(f"  Mismatch! {imgs - lbls} images have no label.")
    print("=" * 50)

inspect_yolo_dataset(DATASET_DIR)


#  STEP 3 — CREATE data.yaml

yaml_content = {
    "path"  : DATASET_DIR,
    "train" : "train/images",
    "val"   : "valid/images",

    "test"  : "test/images",
    "nc"    : len(CLASS_NAMES),
    "names" : CLASS_NAMES
}

yaml_path = os.path.join(DATASET_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

print(f"\n data.yaml created at: {yaml_path}")
print("   Contents:")
with open(yaml_path) as f:
    print(f.read())


#  STEP 4 — TRAIN YOLOv8
print("\n Starting YOLOv8 Training...")
print(f"   Model   : {MODEL_SIZE}")
print(f"   Epochs  : {EPOCHS}")
print(f"   Img size: {IMG_SIZE}")
print(f"   Batch   : {BATCH_SIZE}\n")

model = YOLO(MODEL_SIZE)   # Downloads pretrained weights automatically

results = model.train(
   data    = yaml_path,
    epochs  = EPOCHS,
    imgsz   = IMG_SIZE,
    batch   = BATCH_SIZE,
    project = PROJECT_DIR,
    name    = "bird_drone_detection",
    patience= 10,           # Early stopping if no improvement for 10 epochs
    save    = True,
    plots   = True,         # Auto-generates training plots
    verbose = True
)

print("\n Training complete!")
print(f"   Best model saved at: {PROJECT_DIR}/bird_drone_detection/weights/best.pt")

#  STEP 5 — VALIDATE THE MODEL

print("\n Validating YOLOv8 model on validation set...")

best_model_path = os.path.join(
    PROJECT_DIR, "bird_drone_detection2", "weights", "best.pt"
)

trained_model = YOLO(best_model_path)

val_results = trained_model.val(
    data    = yaml_path,
    imgsz   = IMG_SIZE,
    split   = "val",
    plots   = True
)

print("\n Validation Metrics:")
print(f"   mAP50       : {val_results.box.map50:.4f}")
print(f"   mAP50-95    : {val_results.box.map:.4f}")
print(f"   Precision   : {val_results.box.mp:.4f}")
print(f"   Recall      : {val_results.box.mr:.4f}")


#  STEP 6 — INFERENCE ON TEST IMAGES
print("\n Running inference on test images...")

test_img_dir = os.path.join(DATASET_DIR, "test", "images")
inference_output = os.path.join(PROJECT_DIR, "inference_results")

inference_results = trained_model.predict(
    source     = test_img_dir,
    imgsz      = IMG_SIZE,
    conf       = 0.25,         # Confidence threshold
    iou        = 0.45,         # IoU threshold for NMS
    save       = True,         # Save annotated images
    save_conf  = True,         # Show confidence on boxes
    project    = inference_output,
    name       = "test_predictions"
)

print(f"\n Inference complete!")
print(f"   Annotated images saved at: {inference_output}/test_predictions/")


#  STEP 7 — DISPLAY SAMPLE INFERENCE RESULTS
def show_inference_samples(results_dir, num=6):
    """Display a grid of annotated inference images."""
    pred_dir = os.path.join(results_dir, "test_predictions")
    if not os.path.exists(pred_dir):
        print(" Inference output folder not found.")
        return

    images = [f for f in os.listdir(pred_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))][:num]

    if not images:
        print("No annotated images found.")
        return

    cols = 3
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    fig.suptitle("YOLOv8 — Sample Detection Results", fontsize=14, fontweight="bold")

    for i, img_name in enumerate(images):
        img = mpimg.imread(os.path.join(pred_dir, img_name))
        axes[i].imshow(img)
        axes[i].set_title(img_name, fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("yolo_sample_predictions.png", dpi=150)
    plt.show()
    print(" Sample predictions saved as 'yolo_sample_predictions.png'")

show_inference_samples(inference_output)


#  SUMMARY
print("\n" + "=" * 50)
print("  YOLOv8 PIPELINE COMPLETE")
print("=" * 50)
print(f"  data.yaml        → {yaml_path}")
print(f"  Best model       → {best_model_path}")
print(f"  Training plots   → {PROJECT_DIR}/bird_drone_detection/")
print(f"  Test predictions → {inference_output}/test_predictions/")
print(f"  mAP50 : {val_results.box.map50:.4f}")
print(f"  mAP50-95 : {val_results.box.map:.4f}")

