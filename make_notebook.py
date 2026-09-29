"""
Generator script for fundus_gate_training.ipynb
Creates a clean, robust, fast Jupyter Notebook for Kaggle execution.
"""
import json
import os

cells = []

def add_code_cell(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source.strip()
    })

def add_markdown_cell(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip()
    })

# ==================== CELL 0: Markdown Title ====================
add_markdown_cell("""
# Fundus Gate: Binary Image Classifier (Fundus vs. Non-Fundus)
### Pre-screening Input Validation Gate for Diabetic Retinopathy Diagnostic Pipeline
**Thesis:** Nghiên cứu Vision Transformer và Ứng dụng Phân loại Mức độ Bệnh Tiểu đường từ Ảnh Đáy Võng mạc Mắt  
**Architecture:** Pretrained MobileNetV3Small (Lightweight, Web/Mobile Optimized)  
**Input:** 224x224x3  
**Classes:** `0 = non_fundus`, `1 = fundus`
""")

# ==================== CELL 1: Environment & GPU Setup ====================
add_code_cell("""
import os
import glob
import json
import random
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 2. Hardware & Mixed Precision Configuration
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"GPUs available: {[g.name for g in gpus]}")
    print(f"Mixed Precision Policy: {tf.keras.mixed_precision.global_policy().name}")
else:
    print("WARNING: No GPU detected, running on CPU.")
""")

# ==================== CELL 2: Direct Dataset Discovery (Fast & Deduplicated) ====================
add_code_cell("""
print("=" * 60)
print("1. DIRECT DATASET DISCOVERY & FAST DEDUPLICATION")
print("=" * 60)

start_scan_time = time.time()
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

def get_images_from_dataset(owner_slug, max_images=None):
    candidates = [
        f"/kaggle/input/datasets/{owner_slug}",
        f"/kaggle/input/{owner_slug.split('/')[-1]}",
        f"/kaggle/input/{owner_slug}",
        f"/kaggle/input/competitions/{owner_slug.split('/')[-1]}"
    ]
    ds_root = None
    for c in candidates:
        if os.path.exists(c):
            ds_root = c
            break
            
    if ds_root is None:
        slug = owner_slug.split('/')[-1]
        for root, dirs, _ in os.walk("/kaggle/input"):
            if slug in dirs:
                ds_root = os.path.join(root, slug)
                break
                
    if ds_root is None:
        print(f"  [!] Directory not found for: {owner_slug}")
        return []
        
    imgs = set()
    for root, _, files in os.walk(ds_root):
        for f in files:
            if f.lower().endswith(valid_exts):
                imgs.add(os.path.join(root, f))
                if max_images and len(imgs) >= max_images:
                    return list(imgs)
    print(f"  -> {owner_slug} ({ds_root}): {len(imgs):,} images found")
    return list(imgs)

# 1. Fundus (Positive, EyePACS)
print("Collecting Fundus images...")
fundus_raw = get_images_from_dataset("tanlikesmath/diabetic-retinopathy-resized", max_images=16000)
if len(fundus_raw) == 0:
    fundus_raw = get_images_from_dataset("diabetic-retinopathy-detection", max_images=16000)
fundus_candidates = list(set(fundus_raw))
print(f"Total Unique Fundus: {len(fundus_candidates):,}")

# 2. Group A: Generic Negatives
print("\\nCollecting Generic Negatives...")
generic_natural = get_images_from_dataset("prasunroy/natural-images", max_images=3000)
generic_intel = get_images_from_dataset("puneet6060/intel-image-classification", max_images=3000)
generic_docs = get_images_from_dataset("ritvik1909/document-classification-dataset", max_images=1000)
generic_candidates = list(set(generic_natural + generic_intel + generic_docs))
print(f"Total Unique Generic Negatives: {len(generic_candidates):,}")

# 3. Group B: Medical Negatives (retinal OCT, chest X-ray, brain MRI)
print("\\nCollecting Medical Negatives...")
medical_oct = get_images_from_dataset("shakilrana/octdl-retinal-oct-images-dataset", max_images=3000)
medical_xray = get_images_from_dataset("thomasdubail/chest-pneumonia-256x256", max_images=3000)
medical_mri = get_images_from_dataset("sartajbhuvaji/brain-tumor-classification-mri", max_images=1500)
medical_candidates = list(set(medical_oct + medical_xray + medical_mri))
print(f"Total Unique Medical Negatives: {len(medical_candidates):,}")

# 4. Group C: Ocular Hard Negatives (CASIA iris, eye-dataset close-ups, cataract anterior, eye-disease)
print("\\nCollecting Ocular Hard Negatives...")
ocular_iris = get_images_from_dataset("monareyhanii/casia-iris-syn", max_images=3000)
ocular_eye = get_images_from_dataset("kayvanshah/eye-dataset", max_images=2000)
ocular_cataract = get_images_from_dataset("akshayramakrishnan28/cataract-classification-dataset", max_images=1500)
ocular_diseases = get_images_from_dataset("kondwani/eye-disease-dataset", max_images=1000)
ocular_candidates = list(set(ocular_iris + ocular_eye + ocular_cataract + ocular_diseases))
print(f"Total Unique Ocular Hard Negatives: {len(ocular_candidates):,}")

print(f"\\nDataset scan completed in {time.time() - start_scan_time:.2f}s!")
""")

# ==================== CELL 3: Balance & Challenge Holdout Isolation ====================
add_code_cell("""
print("=" * 60)
print("2. DATASET BALANCING, INTEGRITY & CHALLENGE TEST ISOLATION")
print("=" * 60)

def filter_valid_images(path_list, max_sample=None):
    valid = []
    shuffled = list(set(path_list))
    random.shuffle(shuffled)
    for p in shuffled:
        try:
            if os.path.getsize(p) > 500:
                valid.append(p)
                if max_sample and len(valid) >= max_sample:
                    break
        except Exception:
            continue
    return valid

TARGET_PER_GROUP = 3500
valid_generic = filter_valid_images(generic_candidates, TARGET_PER_GROUP)
valid_medical = filter_valid_images(medical_candidates, TARGET_PER_GROUP)
valid_ocular_all = filter_valid_images(ocular_candidates, TARGET_PER_GROUP + 1000)

print(f"Valid pool: Generic={len(valid_generic):,}, Medical={len(valid_medical):,}, Ocular={len(valid_ocular_all):,}")

# 1. STRICT ISOLATION OF OCULAR HARD NEGATIVES CHALLENGE TEST
CHALLENGE_SIZE = min(1000, max(500, int(len(valid_ocular_all) * 0.20)))
random.shuffle(valid_ocular_all)
challenge_ocular_paths = valid_ocular_all[:CHALLENGE_SIZE]
standard_ocular_paths = valid_ocular_all[CHALLENGE_SIZE:CHALLENGE_SIZE + TARGET_PER_GROUP]

challenge_test_df = pd.DataFrame({
    'filepath': challenge_ocular_paths,
    'label': 0,
    'category': 'ocular_hard_challenge',
    'patient_id': ['non_fundus'] * len(challenge_ocular_paths)
})
print(f"Strictly Isolated Ocular Challenge Test Set: {len(challenge_test_df):,} images")
print("  (Zero exposure during model training or validation)")

# 2. Build Standard Negatives
generic_df = pd.DataFrame({
    'filepath': valid_generic,
    'label': 0,
    'category': 'generic',
    'patient_id': ['non_fundus'] * len(valid_generic)
})

medical_df = pd.DataFrame({
    'filepath': valid_medical,
    'label': 0,
    'category': 'medical',
    'patient_id': ['non_fundus'] * len(valid_medical)
})

ocular_df = pd.DataFrame({
    'filepath': standard_ocular_paths,
    'label': 0,
    'category': 'ocular_hard',
    'patient_id': ['non_fundus'] * len(standard_ocular_paths)
})

total_negatives_count = len(generic_df) + len(medical_df) + len(ocular_df)
print(f"Total Standard Non-Fundus Pool: {total_negatives_count:,} images")

# 3. Sample Equal Number of Fundus Images for 50/50 Balance
TARGET_FUNDUS = total_negatives_count
valid_fundus = filter_valid_images(fundus_candidates, TARGET_FUNDUS)

fundus_df = pd.DataFrame({
    'filepath': valid_fundus,
    'label': 1,
    'category': 'fundus'
})
fundus_df['patient_id'] = fundus_df['filepath'].apply(
    lambda x: os.path.basename(x).split('.')[0].split('_')[0]
)

print(f"Standard Fundus Pool: {len(fundus_df):,} images ({fundus_df['patient_id'].nunique():,} unique patients)")
print(f"Balanced Dataset Ratio: {len(fundus_df)/(len(fundus_df) + total_negatives_count)*100:.1f}% Fundus / {total_negatives_count/(len(fundus_df) + total_negatives_count)*100:.1f}% Non-Fundus")
""")

# ==================== CELL 4: Split 70/15/15 & Zero-Leakage Verification ====================
add_code_cell("""
print("=" * 60)
print("3. SPLIT TRAIN (70%) / VAL (15%) / TEST (15%) & DATA INTEGRITY VERIFICATION")
print("=" * 60)

# 1. Split Fundus using GroupShuffleSplit on patient_id to prevent data leakage
# Train: 70%, Val: 15%, Test: 15%
gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
train_idx, temp_idx = next(gss_train.split(fundus_df, groups=fundus_df['patient_id']))
fundus_train = fundus_df.iloc[train_idx].copy()
fundus_temp = fundus_df.iloc[temp_idx].copy()

# Split remaining 30% equally into Val (15%) and Test (15%)
gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss_val.split(fundus_temp, groups=fundus_temp['patient_id']))
fundus_val = fundus_temp.iloc[val_idx].copy()
fundus_test = fundus_temp.iloc[test_idx].copy()

# Verify zero patient overlap in fundus
p_train = set(fundus_train['patient_id'])
p_val = set(fundus_val['patient_id'])
p_test = set(fundus_test['patient_id'])
assert len(p_train.intersection(p_val)) == 0, "LEAKAGE DETECTED between Train and Val!"
assert len(p_train.intersection(p_test)) == 0, "LEAKAGE DETECTED between Train and Test!"
assert len(p_val.intersection(p_test)) == 0, "LEAKAGE DETECTED between Val and Test!"
print("Zero patient leakage verified across Fundus Train, Val, and Test.")

# 2. Split Negatives with Stratification on Category (Generic, Medical, Ocular)
negatives_df = pd.concat([generic_df, medical_df, ocular_df], ignore_index=True)
negatives_df = negatives_df.drop_duplicates(subset=['filepath']).reset_index(drop=True)

neg_train, neg_temp = train_test_split(
    negatives_df, train_size=0.70, stratify=negatives_df['category'], random_state=SEED
)
neg_val, neg_test = train_test_split(
    neg_temp, train_size=0.50, stratify=neg_temp['category'], random_state=SEED
)

# 3. Combine into final Train, Val, and Test DataFrames
train_df = pd.concat([fundus_train, neg_train], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_df = pd.concat([fundus_val, neg_val], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
test_df = pd.concat([fundus_test, neg_test], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# 4. Strict Filepath Disjointness Verification
all_train_paths = set(train_df['filepath'])
all_val_paths = set(val_df['filepath'])
all_test_paths = set(test_df['filepath'])
all_challenge_paths = set(challenge_test_df['filepath'])

assert len(all_train_paths.intersection(all_val_paths)) == 0, "Train-Val overlap detected!"
assert len(all_train_paths.intersection(all_test_paths)) == 0, "Train-Test overlap detected!"
assert len(all_val_paths.intersection(all_test_paths)) == 0, "Val-Test overlap detected!"
assert len(all_challenge_paths.intersection(all_train_paths)) == 0, "Challenge-Train overlap detected!"
assert len(all_challenge_paths.intersection(all_val_paths)) == 0, "Challenge-Val overlap detected!"
assert len(all_challenge_paths.intersection(all_test_paths)) == 0, "Challenge-Test overlap detected!"
print("Zero image overlap verified between all partitions including Challenge Test.")

# 5. Display Breakdown Table
split_summary = pd.DataFrame({
    'Train (70%)': train_df['category'].value_counts(),
    'Val (15%)': val_df['category'].value_counts(),
    'Test (15%)': test_df['category'].value_counts()
}).fillna(0).astype(int)
split_summary.loc['TOTAL'] = split_summary.sum()
print("\\n--- SPLIT COMPOSITION SUMMARY ---")
print(split_summary.to_string())
print(f"\\nChallenge Test (Ocular Hard Holdout): {len(challenge_test_df):,} images")
""")

# ==================== CELL 5: tf.data Pipeline & Light Augmentation ====================
add_code_cell("""
print("=" * 60)
print("4. BUILDING OPTIMIZED TF.DATA PIPELINE WITH LIGHT AUGMENTATION")
print("=" * 60)

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    label = tf.cast(label, tf.float32)
    return img, label

# Light augmentation layer: horizontal flip, slight rotation, zoom, contrast
augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor=0.05),     # ~18 degrees
    layers.RandomZoom(height_factor=0.05, width_factor=0.05),
    layers.RandomContrast(factor=0.1)
], name="light_augmentation")

def augment_image(img, label):
    img = augmentation_layer(img, training=True)
    return img, label

def create_tf_dataset(df, is_training=False):
    paths = df['filepath'].values
    labels = df['label'].values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
        
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = create_tf_dataset(train_df, is_training=True)
val_ds = create_tf_dataset(val_df, is_training=False)
test_ds = create_tf_dataset(test_df, is_training=False)
challenge_ds = create_tf_dataset(challenge_test_df, is_training=False)

# Sanity check: inspect first batch
sample_batch_imgs, sample_batch_labels = next(iter(train_ds))
print(f"Batch Image Shape: {sample_batch_imgs.shape} (Expected: ({BATCH_SIZE}, 224, 224, 3))")
print(f"Batch Label Shape: {sample_batch_labels.shape}")
print(f"Batch Label Distribution: {int(tf.reduce_sum(sample_batch_labels).numpy())} Fundus, {BATCH_SIZE - int(tf.reduce_sum(sample_batch_labels).numpy())} Non-Fundus")

# Plot 6 sanity check samples
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i in range(6):
    img = sample_batch_imgs[i].numpy().astype(np.float32)
    img_disp = (img + 1.0) / 2.0
    lbl = int(sample_batch_labels[i].numpy())
    axes[i].imshow(np.clip(img_disp, 0.0, 1.0).astype(np.float32))
    axes[i].set_title(f"{'FUNDUS (1)' if lbl==1 else 'NON-FUNDUS (0)'}")
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('/kaggle/working/data_sanity_check.png', dpi=150)
plt.close(fig)
print("Saved data sanity check plot to /kaggle/working/data_sanity_check.png")
""")

# ==================== CELL 6: MobileNetV3Small Architecture ====================
add_code_cell("""
print("=" * 60)
print("5. BUILDING MOBILENETV3SMALL CLASSIFIER")
print("=" * 60)

def build_fundus_gate_model():
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Initially frozen for Stage 1
    
    inputs = layers.Input(shape=(224, 224, 3), name='input_image')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout_regularization')(x)
    # Explicit float32 dtype on final output layer for numerical stability in mixed precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='fundus_gate_prob')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='FundusGate_MobileNetV3Small')
    return model, base_model

model, base_model = build_fundus_gate_model()
model.summary()

trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
non_trainable_params = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)
print(f"Total Parameters: {trainable_params + non_trainable_params:,}")
print(f"Trainable Parameters (Stage 1): {trainable_params:,}")
print(f"Frozen Parameters: {non_trainable_params:,}")
""")

# ==================== CELL 7: Stage 1 Training (Frozen Head) ====================
add_code_cell("""
print("=" * 60)
print("6. STAGE 1 TRAINING: CLASSIFICATION HEAD ONLY (FROZEN BACKBONE)")
print("=" * 60)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='roc_auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

STAGE1_EPOCHS = 3
start_time = time.time()

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    verbose=1
)

print(f"Stage 1 completed in {time.time() - start_time:.1f}s")
""")

# ==================== CELL 8: Stage 2 Training (Unfreeze Top Layers) ====================
add_code_cell("""
print("=" * 60)
print("7. STAGE 2 TRAINING: FINE-TUNING TOP 30% LAYERS")
print("=" * 60)

# Unfreeze top ~30% layers of MobileNetV3Small
base_model.trainable = True
num_layers = len(base_model.layers)
unfreeze_from = int(num_layers * 0.70)  # Freeze first 70%, unfreeze last 30%

for layer in base_model.layers[:unfreeze_from]:
    layer.trainable = False
for layer in base_model.layers[unfreeze_from:]:
    layer.trainable = True

print(f"Total layers in MobileNetV3Small: {num_layers}")
print(f"Frozen layers: 0 to {unfreeze_from-1}")
print(f"Unfrozen fine-tuning layers: {unfreeze_from} to {num_layers-1} ({num_layers - unfreeze_from} layers)")

# Recompile with small learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='roc_auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

STAGE2_EPOCHS = 3
best_model_path = '/kaggle/working/fundus_gate_best.keras'

training_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    )
]

start_time = time.time()
history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    callbacks=training_callbacks,
    verbose=1
)
print(f"Stage 2 fine-tuning completed in {time.time() - start_time:.1f}s")

# Load best checkpoint
if os.path.exists(best_model_path):
    print(f"Loading best weights from {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)
""")

# ==================== CELL 9: Validation-Based Threshold Tuning ====================
add_code_cell("""
print("=" * 60)
print("8. VALIDATION SET THRESHOLD OPTIMIZATION")
print("=" * 60)

# Compute predictions on Validation Set
print("Generating predictions on Validation Set...")
val_preds = model.predict(val_ds, verbose=1).flatten()
val_labels = val_df['label'].values

# Calculate ROC curve & AUC
val_auc = roc_auc_score(val_labels, val_preds)
print(f"Validation ROC-AUC: {val_auc:.4f}")

# Threshold Search:
# Objective: Fundus Recall >= 95% while minimizing False Positive Rate (FPR) / false acceptance
candidate_thresholds = np.linspace(0.01, 0.99, 197)
threshold_records = []

for t in candidate_thresholds:
    preds_binary = (val_preds >= t).astype(int)
    rec = recall_score(val_labels, preds_binary, zero_division=0)
    prec = precision_score(val_labels, preds_binary, zero_division=0)
    f1 = f1_score(val_labels, preds_binary, zero_division=0)
    
    # Specificity & FPR
    tn, fp, fn, tp = confusion_matrix(val_labels, preds_binary).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    
    threshold_records.append({
        'threshold': float(t),
        'recall': float(rec),
        'precision': float(prec),
        'f1': float(f1),
        'specificity': float(spec),
        'fpr': float(fpr)
    })

thresh_df = pd.DataFrame(threshold_records)

# Filter thresholds satisfying Fundus Recall >= 95% (0.950)
valid_candidates = thresh_df[thresh_df['recall'] >= 0.950]

if len(valid_candidates) > 0:
    # Select threshold that minimizes FPR (i.e. minimizes false acceptance)
    sorted_candidates = valid_candidates.sort_values(by=['fpr', 'f1'], ascending=[True, False])
    optimal_row = sorted_candidates.iloc[0]
    best_thresh = float(optimal_row['threshold'])
    print(f"Selected Optimal Threshold: {best_thresh:.4f}")
    print(f"  Val Recall: {optimal_row['recall']*100:.2f}% (>= 95% constraint satisfied)")
    print(f"  Val Specificity: {optimal_row['specificity']*100:.2f}%")
    print(f"  Val FPR (False Acceptance Rate): {optimal_row['fpr']*100:.2f}%")
    print(f"  Val F1-Score: {optimal_row['f1']:.4f}")
else:
    # Fallback to Youden's J index
    thresh_df['youden_j'] = thresh_df['recall'] + thresh_df['specificity'] - 1.0
    best_thresh = float(thresh_df.loc[thresh_df['youden_j'].idxmax()]['threshold'])
    print(f"Fallback Optimal Threshold: {best_thresh:.4f}")

# Save threshold.json
val_row_chosen = thresh_df.iloc[(thresh_df['threshold'] - best_thresh).abs().argsort()[:1]].iloc[0]
threshold_meta = {
    'optimal_threshold': round(best_thresh, 4),
    'constraint': 'Fundus Recall >= 0.95 and minimum FPR',
    'validation_metrics': {
        'roc_auc': round(float(val_auc), 4),
        'recall': round(float(val_row_chosen['recall']), 4),
        'specificity': round(float(val_row_chosen['specificity']), 4),
        'fpr': round(float(val_row_chosen['fpr']), 4),
        'f1': round(float(val_row_chosen['f1']), 4)
    }
}

with open('/kaggle/working/threshold.json', 'w') as f:
    json.dump(threshold_meta, f, indent=4)
print("Saved threshold metadata to /kaggle/working/threshold.json")

# Plot Threshold Optimization Curves
plt.figure(figsize=(10, 6))
plt.plot(thresh_df['threshold'], thresh_df['recall'], label='Fundus Recall (Sensitivity)', color='blue', lw=2)
plt.plot(thresh_df['threshold'], thresh_df['specificity'], label='Specificity (1 - FPR)', color='green', lw=2)
plt.plot(thresh_df['threshold'], thresh_df['fpr'], label='FPR (False Acceptance)', color='red', lw=2, linestyle='--')
plt.axvline(best_thresh, color='black', linestyle=':', label=f'Chosen Threshold = {best_thresh:.2f}', lw=2)
plt.axhline(0.95, color='gray', linestyle='-.', label='Target Recall >= 95%')
plt.title('Threshold Optimization on Validation Set', fontsize=14, fontweight='bold')
plt.xlabel('Classification Threshold', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.savefig('/kaggle/working/threshold_tuning.png', dpi=150)
plt.close('all')
""")

# ==================== CELL 10: Test Set Evaluation ====================
add_code_cell("""
print("=" * 60)
print(f"9. EVALUATION ON UNSEEN TEST SET (FIXED THRESHOLD = {best_thresh:.4f})")
print("=" * 60)

test_preds = model.predict(test_ds, verbose=1).flatten()
test_labels = test_df['label'].values
test_preds_binary = (test_preds >= best_thresh).astype(int)

# Comprehensive Metrics
test_acc = accuracy_score(test_labels, test_preds_binary)
test_prec = precision_score(test_labels, test_preds_binary, zero_division=0)
test_rec = recall_score(test_labels, test_preds_binary, zero_division=0)
test_f1 = f1_score(test_labels, test_preds_binary, zero_division=0)
test_auc = roc_auc_score(test_labels, test_preds)

tn, fp, fn, tp = confusion_matrix(test_labels, test_preds_binary).ravel()
test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
test_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
test_fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

print(f"Test Accuracy:    {test_acc*100:.2f}%")
print(f"Test Precision:   {test_prec*100:.2f}%")
print(f"Test Recall:      {test_rec*100:.2f}% (Sensitivity)")
print(f"Test Specificity: {test_spec*100:.2f}% (True Negative Rate)")
print(f"Test FPR:         {test_fpr*100:.2f}% (False Acceptance Rate)")
print(f"Test FNR:         {test_fnr*100:.2f}%")
print(f"Test F1-Score:    {test_f1:.4f}")
print(f"Test ROC-AUC:     {test_auc:.4f}")
print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# Subgroup Breakdown on Test Set
test_df['predicted_prob'] = test_preds
test_df['predicted_class'] = test_preds_binary

print("\\n--- FALSE ACCEPTANCE RATE (FPR) BY SUBGROUP ---")
for cat in ['generic', 'medical', 'ocular_hard']:
    sub_df = test_df[test_df['category'] == cat]
    if len(sub_df) > 0:
        sub_fp = (sub_df['predicted_class'] == 1).sum()
        sub_fpr = sub_fp / len(sub_df)
        print(f"  {cat.capitalize()} Negatives: {sub_fp}/{len(sub_df)} false accepted (FPR = {sub_fpr*100:.2f}%)")

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Non-Fundus (0)', 'Pred Fundus (1)'],
            yticklabels=['Actual Non-Fundus (0)', 'Actual Fundus (1)'])
plt.title(f'Test Confusion Matrix (Threshold={best_thresh:.2f})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.close('all')
""")

# ==================== CELL 11: Challenge Test Evaluation ====================
add_code_cell("""
print("=" * 60)
print(f"10. EVALUATION ON OCULAR HARD NEGATIVES CHALLENGE TEST (STRICT HOLDOUT)")
print("=" * 60)

challenge_preds = model.predict(challenge_ds, verbose=1).flatten()
challenge_preds_binary = (challenge_preds >= best_thresh).astype(int)

challenge_test_df['predicted_prob'] = challenge_preds
challenge_test_df['predicted_class'] = challenge_preds_binary

total_challenge = len(challenge_test_df)
challenge_fp = int((challenge_preds_binary == 1).sum())
challenge_tn = int((challenge_preds_binary == 0).sum())
challenge_fpr = challenge_fp / total_challenge if total_challenge > 0 else 0.0
challenge_spec = challenge_tn / total_challenge if total_challenge > 0 else 0.0

print(f"Total Challenge Hard Ocular Samples: {total_challenge:,}")
print(f"Correctly Rejected (TN):             {challenge_tn:,} ({challenge_spec*100:.2f}%)")
print(f"False Acceptance (FP):               {challenge_fp:,} ({challenge_fpr*100:.2f}%)")
print(f"Challenge Specificity:               {challenge_spec*100:.2f}%")
print(f"Challenge FPR (False Acceptance):    {challenge_fpr*100:.2f}%")
""")

# ==================== CELL 12: Error Analysis ====================
add_code_cell("""
print("=" * 60)
print("11. ERROR ANALYSIS: INSPECTING MISCLASSIFIED SAMPLES")
print("=" * 60)

fps_standard = test_df[(test_df['label'] == 0) & (test_df['predicted_class'] == 1)]
fps_challenge = challenge_test_df[(challenge_test_df['label'] == 0) & (challenge_test_df['predicted_class'] == 1)]
all_fps = pd.concat([fps_standard, fps_challenge], ignore_index=True)

all_fns = test_df[(test_df['label'] == 1) & (test_df['predicted_class'] == 0)]

print(f"Total False Positives in Test + Challenge: {len(all_fps)}")
print(f"Total False Negatives in Test: {len(all_fns)}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Top row: False Positives
axes[0, 0].set_ylabel("False Positives\\n(Non-Fundus → Accepted)", fontsize=11, fontweight='bold')
for i in range(4):
    ax = axes[0, i]
    if i < len(all_fps):
        row = all_fps.iloc[i]
        try:
            img = Image.open(row['filepath']).convert('RGB').resize((224, 224))
            ax.imshow(np.array(img, dtype=np.uint8))
            ax.set_title(f"Cat: {row['category']}\\nP(fundus)={row['predicted_prob']:.2f}", fontsize=10, color='red')
        except Exception:
            ax.text(0.5, 0.5, "Load Error", ha='center')
    else:
        ax.text(0.5, 0.5, "No Error", ha='center')
    ax.axis('off')

# Bottom row: False Negatives
axes[1, 0].set_ylabel("False Negatives\\n(Fundus → Rejected)", fontsize=11, fontweight='bold')
for i in range(4):
    ax = axes[1, i]
    if i < len(all_fns):
        row = all_fns.iloc[i]
        try:
            img = Image.open(row['filepath']).convert('RGB').resize((224, 224))
            ax.imshow(np.array(img, dtype=np.uint8))
            ax.set_title(f"Actual: Fundus\\nP(fundus)={row['predicted_prob']:.2f}", fontsize=10, color='orange')
        except Exception:
            ax.text(0.5, 0.5, "Load Error", ha='center')
    else:
        ax.text(0.5, 0.5, "No Error", ha='center')
    ax.axis('off')

plt.suptitle(f"Error Analysis on False Acceptances & False Rejections (Threshold={best_thresh:.2f})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/error_analysis.png', dpi=150)
plt.close('all')
print("Saved error analysis plot to /kaggle/working/error_analysis.png")
""")

# ==================== CELL 13: Model Export & Format Conversions ====================
add_code_cell("""
print("=" * 60)
print("12. EXPORTING FINAL MODEL ARTIFACTS (KERAS, TFLITE, TFLITE-FP16)")
print("=" * 60)

export_dir = "/kaggle/working"
keras_path = os.path.join(export_dir, "fundus_gate.keras")
tflite_path = os.path.join(export_dir, "fundus_gate.tflite")
tflite_fp16_path = os.path.join(export_dir, "fundus_gate_float16.tflite")

# 1. Save Full Keras Model
model.save(keras_path)
print(f"Saved Keras model: {keras_path} ({os.path.getsize(keras_path)/1024/1024:.2f} MB)")

# 2. Convert to Standard TFLite (Float32) & Float16 Quantized via CPU Subprocess
# (Prevents multi-GPU Grappler session deadlock on Dual Tesla T4 VMs)
convert_script = "\\n".join([
    "import os, sys",
    "os.environ['CUDA_VISIBLE_DEVICES'] = ''",
    "import tensorflow as tf",
    "",
    f"keras_path = '{keras_path}'",
    f"tflite_path = '{tflite_path}'",
    f"tflite_fp16_path = '{tflite_fp16_path}'",
    "",
    "print('Loading Keras model on CPU for clean TFLite conversion...')",
    "model = tf.keras.models.load_model(keras_path)",
    "run_model = tf.function(lambda x: model(x))",
    "concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 224, 224, 3], tf.float32))",
    "",
    "print('Converting to Float32 TFLite...')",
    "converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])",
    "tflite_model = converter.convert()",
    "with open(tflite_path, 'wb') as f:",
    "    f.write(tflite_model)",
    "print(f'Saved TFLite model: {tflite_path} ({os.path.getsize(tflite_path)/1024/1024:.2f} MB)')",
    "",
    "print('Converting to Float16 Quantized TFLite...')",
    "converter_fp16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])",
    "converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]",
    "converter_fp16.target_spec.supported_types = [tf.float16]",
    "tflite_fp16_model = converter_fp16.convert()",
    "with open(tflite_fp16_path, 'wb') as f:",
    "    f.write(tflite_fp16_model)",
    "print(f'Saved TFLite Float16 model: {tflite_fp16_path} ({os.path.getsize(tflite_fp16_path)/1024/1024:.2f} MB)')",
    "print('TFLite CPU Conversion Complete!')"
])

script_path = "/kaggle/working/convert_tflite.py"
with open(script_path, "w") as f:
    f.write(convert_script)

import sys, subprocess
proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
print(proc.stdout)
if proc.stderr:
    print(proc.stderr)
assert proc.returncode == 0, f"TFLite conversion failed with returncode {proc.returncode}"

# 4. Verify TFLite Inference
print("\\nVerifying TFLite Inference...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

dummy_input = np.random.uniform(-1.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
dummy_pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
print(f"TFLite Dummy Inference Output: {dummy_pred:.4f} (Valid Range [0, 1])")

# 5. Save Complete metrics.json
full_metrics = {
    'model_name': 'FundusGate_MobileNetV3Small',
    'input_shape': [224, 224, 3],
    'total_parameters': int(model.count_params()),
    'optimal_threshold': round(best_thresh, 4),
    'validation_metrics': {
        'roc_auc': round(float(val_auc), 4),
        'recall': round(float(val_row_chosen['recall']), 4),
        'specificity': round(float(val_row_chosen['specificity']), 4),
        'fpr': round(float(val_row_chosen['fpr']), 4),
        'f1': round(float(val_row_chosen['f1']), 4)
    },
    'test_metrics': {
        'accuracy': round(float(test_acc), 4),
        'precision': round(float(test_prec), 4),
        'recall': round(float(test_rec), 4),
        'specificity': round(float(test_spec), 4),
        'fpr': round(float(test_fpr), 4),
        'fnr': round(float(test_fnr), 4),
        'f1': round(float(test_f1), 4),
        'roc_auc': round(float(test_auc), 4),
        'confusion_matrix': {
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }
    },
    'challenge_test_metrics': {
        'total_samples': total_challenge,
        'true_negatives': challenge_tn,
        'false_positives': challenge_fp,
        'challenge_specificity': round(float(challenge_spec), 4),
        'challenge_fpr': round(float(challenge_fpr), 4)
    },
    'artifacts': {
        'keras_size_mb': round(os.path.getsize(keras_path)/1024/1024, 2),
        'tflite_size_mb': round(os.path.getsize(tflite_path)/1024/1024, 2),
        'tflite_fp16_size_mb': round(os.path.getsize(tflite_fp16_path)/1024/1024, 2)
    }
}

with open('/kaggle/working/metrics.json', 'w') as f:
    json.dump(full_metrics, f, indent=4)
print("Saved full metrics to /kaggle/working/metrics.json")

# 6. Final Console Summary Report
print("\\n" + "=" * 60)
print("FINAL STANDARDIZED REPORT BLOCK")
print("=" * 60)
print(f"Dataset: EyePACS ({len(fundus_df):,} Fundus) vs 3-Group Negatives ({len(negatives_df):,} Non-Fundus)")
print(f"Model: MobileNetV3Small (Pretrained ImageNet)")
print(f"Parameters: {model.count_params():,}")
print(f"Threshold: {best_thresh:.4f} (Calibrated on Validation set for Recall >= 95%)")
print(f"Test Recall: {test_rec*100:.2f}%")
print(f"Test Specificity: {test_spec*100:.2f}%")
print(f"Test FPR: {test_fpr*100:.2f}%")
print(f"Challenge FPR: {challenge_fpr*100:.2f}%")
print(f"Model size: {os.path.getsize(keras_path)/1024/1024:.2f} MB")
print(f"TFLite size: {os.path.getsize(tflite_path)/1024/1024:.2f} MB (FP16: {os.path.getsize(tflite_fp16_path)/1024/1024:.2f} MB)")
print("=" * 60)
""")

# Build Notebook dict
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "dockerImageVersionId": 30699,
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), "fundus_gate_training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Generated {out_path} with {len(cells)} cells successfully!")
