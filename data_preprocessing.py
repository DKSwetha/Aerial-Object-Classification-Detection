
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. CONFIGURATION
DATASET_DIR = r"D:\Labmentix\aerial object\classification_dataset"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR  = os.path.join(DATASET_DIR, "test")


#  2. INSPECT DATASET
def inspect_dataset(dataset_dir):
    print("DATASET INSPECTION")
    for split in ["TRAIN", "VALID", "TEST"]:
        split_path = os.path.join(dataset_dir, split)
        print(f"\n {split}")
        total = 0
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f"   └── {cls}: {count} images")
                total += count
        print(f"   Total: {total} images")
    print("=" * 45)

inspect_dataset(DATASET_DIR)


#  3. VISUALIZE SAMPLE IMAGES
def visualize_samples(dataset_dir, split="TRAIN", num_samples=6):
    split_path = os.path.join(dataset_dir, split)
    classes = sorted(os.listdir(split_path))
    
    fig, axes = plt.subplots(len(classes), num_samples,
                              figsize=(num_samples * 2.5, len(classes) * 2.5))
    fig.suptitle(f"Sample Images — {split} Set", fontsize=14, fontweight="bold")

    for row, cls in enumerate(classes):
        cls_path = os.path.join(split_path, cls)
        images = os.listdir(cls_path)[:num_samples]
        for col, img_name in enumerate(images):
            img_path = os.path.join(cls_path, img_name)
            img = plt.imread(img_path)
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_title(cls.upper(), fontsize=11,
                                          fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.show()
    print(" Sample image grid saved as 'sample_images.png'")

visualize_samples(DATASET_DIR)


#  4. DATA AUGMENTATION (TRAIN ONLY)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixel values to [0, 1]
    rotation_range=20,           # Randomly rotate images up to 20°
    width_shift_range=0.1,       # Randomly shift horizontally
    height_shift_range=0.1,      # Randomly shift vertically
    zoom_range=0.15,             # Random zoom in/out
    horizontal_flip=True,        # Flip left-right
    brightness_range=[0.8, 1.2], # Random brightness adjustment
    fill_mode="nearest"          # Fill empty pixels after transforms
)

# Validation & Test — ONLY rescale, no augmentation
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)


#  5. CREATE DATA GENERATORS (Loaders)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",    # Bird=0, Drone=1 (binary classification)
    seed=SEED,
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    seed=SEED,
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    seed=SEED,
    shuffle=False
)


#  6. VERIFY GENERATORS
print("\n Data Generators Ready")
print(f"   Class mapping : {train_generator.class_indices}")
print(f"   Train batches : {len(train_generator)}")
print(f"   Valid batches : {len(val_generator)}")
print(f"   Test  batches : {len(test_generator)}")


#  7. VISUALIZE AUGMENTED SAMPLES
def visualize_augmented(generator, num_images=8):
    images, labels = next(generator)
    class_names = {v: k for k, v in generator.class_indices.items()}

    fig, axes = plt.subplots(2, num_images // 2, figsize=(num_images * 2, 5))
    axes = axes.flatten()
    fig.suptitle("Augmented Training Samples", fontsize=13, fontweight="bold")

    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title(class_names[int(labels[i])], fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("augmented_samples.png", dpi=150)
    plt.show()
    print(" Augmented samples saved as 'augmented_samples.png'")

visualize_augmented(train_generator)
