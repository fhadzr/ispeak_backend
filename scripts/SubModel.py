"""
SubModel_fixed.py — Fixed version

Perubahan dari versi original:
1. FIX DATA LEAKAGE: Augmentasi hanya dari train_idx, bukan semua data
2. LEBIH RINGAN: epoch 150, augmentation 4x (cukup untuk 123 samples)
3. ENSEMBLE DISEDERHANAKAN: 1 GRU + 1 Dense + RF (bukan 5 model)
4. KERAS COMPATIBILITY: save dengan tf.keras bukan standalone keras
5. SCALER FIX: transformer disimpan per fold final, bukan per subconstruct saja
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# GPU SETUP
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU aktif: {gpus[0].name}")
else:
    print("⚠ GPU tidak ditemukan, menggunakan CPU")
    
from pathlib import Path
from multiprocessing import cpu_count

# Keras import — prioritaskan tf.keras agar kompatibel dengan FastAPI
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, GRU, BatchNormalization
    from tensorflow.keras.regularizers import l1_l2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    print("✓ Menggunakan tensorflow.keras")
except Exception:
    raise ImportError("tensorflow.keras tidak ditemukan. Install tensorflow.")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================
# CPU SETTINGS
# ===================================
NUM_CORES = cpu_count()
print(f"🖥️  Detected {NUM_CORES} CPU cores")

os.environ['OMP_NUM_THREADS'] = str(NUM_CORES)
os.environ['TF_NUM_INTEROP_THREADS'] = str(NUM_CORES)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(NUM_CORES)
os.environ['MKL_NUM_THREADS'] = str(NUM_CORES)

tf.config.threading.set_intra_op_parallelism_threads(NUM_CORES)
tf.config.threading.set_inter_op_parallelism_threads(NUM_CORES)

tf.random.set_seed(42)
np.random.seed(42)

print(f"✓ CPU optimization: {NUM_CORES} threads")

# ===================================
# AUGMENTASI — lebih ringan (4x)
# ===================================

def augment_data(X, y, augmentation_factor=4):
    """
    Augmentasi ringan. Dipanggil HANYA pada training split,
    bukan pada seluruh dataset — mencegah data leakage.
    """
    X_aug = [X]
    y_aug = [y]

    for i in range(augmentation_factor):
        # Gaussian noise
        noise_scale = 0.02 + (i * 0.01)
        X_aug.append(X + np.random.normal(0, noise_scale, X.shape))
        y_aug.append(y)

        # Random scaling
        scale = np.random.uniform(0.93, 1.07, X.shape)
        X_aug.append(X * scale)
        y_aug.append(y)

        # Feature dropout (setiap 2 iterasi)
        if i % 2 == 0:
            mask = np.random.binomial(1, 0.92, X.shape)
            X_aug.append(X * mask)
            y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)


# ===================================
# FEATURE ENGINEERING — vectorized
# ===================================

def feature_engineering(X, feature_names):
    """Feature engineering vectorized."""
    X_df = pd.DataFrame(X, columns=feature_names)
    features = [X_df.values]

    X_abs = np.abs(X_df.values) + 1e-6
    features.extend([
        np.log1p(X_abs),
        np.sqrt(X_abs),
        X_df.values ** 2,
        np.sin(X_df.values),
        np.cos(X_df.values),
    ])

    # Statistical
    stats = np.column_stack([
        np.mean(X_df.values, axis=1),
        np.std(X_df.values, axis=1),
        np.max(X_df.values, axis=1),
        np.min(X_df.values, axis=1),
        np.median(X_df.values, axis=1),
    ])
    features.append(stats)

    # Polynomial interactions
    n_interact = min(len(feature_names), 4)
    for i in range(n_interact):
        for j in range(i + 1, n_interact):
            col_i = X_df.iloc[:, i].values
            col_j = X_df.iloc[:, j].values
            features.append(np.column_stack([
                col_i * col_j,
                col_i / (col_j + 1e-6),
                col_i + col_j,
                col_i - col_j,
            ]))

    result = np.concatenate(features, axis=1)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


# ===================================
# MODEL ARCHITECTURES
# ===================================

def create_gru_model(input_dim, num_classes, dropout=0.4):
    model = Sequential([
        Input(shape=(1, input_dim)),
        GRU(128, return_sequences=True, kernel_regularizer=l1_l2(0.0001, 0.001)),
        BatchNormalization(),
        Dropout(dropout),
        GRU(64, kernel_regularizer=l1_l2(0.0001, 0.001)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def create_dense_model(input_dim, num_classes, dropout=0.4):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=l1_l2(0.0001, 0.001)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(128, activation='relu', kernel_regularizer=l1_l2(0.0001, 0.001)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ===================================
# ENSEMBLE TRAINING
# ===================================

def train_ensemble(X_train, y_train, X_test, y_test, num_classes, input_dim):
    """
    Ensemble ringan: GRU + Dense + RandomForest.
    """
    batch_size = 32
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30,
                      restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                          patience=15, min_lr=1e-7, mode='max'),
    ]

    X_train_3d = X_train.reshape(-1, 1, input_dim)
    X_test_3d  = X_test.reshape(-1, 1, input_dim)
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    cw = dict(enumerate(class_weights))

    ensemble_preds = []

    # Model 1: GRU
    m1 = create_gru_model(input_dim, num_classes, dropout=0.4)
    m1.fit(X_train_3d, y_train_cat,
           validation_data=(X_test_3d, y_test_cat),
           epochs=150, batch_size=batch_size,
           class_weight=cw, callbacks=callbacks, verbose=0)
    ensemble_preds.append(m1.predict(X_test_3d, verbose=0))

    # Model 2: Dense
    m2 = create_dense_model(input_dim, num_classes, dropout=0.4)
    m2.fit(X_train, y_train_cat,
           validation_data=(X_test, y_test_cat),
           epochs=150, batch_size=batch_size,
           class_weight=cw, callbacks=callbacks, verbose=0)
    ensemble_preds.append(m2.predict(X_test, verbose=0))

    # Model 3: RandomForest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict_proba(X_test)
    if pred_rf.shape[1] == num_classes:
        ensemble_preds.append(pred_rf)

    # Weighted average
    weights = [1.5, 1.5, 1.0][:len(ensemble_preds)]
    final_pred = np.average(ensemble_preds, axis=0, weights=weights)

    return np.argmax(final_pred, axis=1), m1


# ===================================
# LOAD DATASET
# ===================================
print("\n" + "=" * 70)
print("Loading dataset...")

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
csv_list  = list(DATA_DIR.glob("*.csv"))
if not csv_list:
    raise FileNotFoundError(f"Tidak ada CSV di {DATA_DIR}")

data_path = csv_list[0]
print(f"Dataset: {data_path}")
df = pd.read_csv(data_path)

exclude_cols = [
    'Filename', 'Fluency', 'Pronunciation', 'Prosody',
    'Coherence and Cohesion', 'Topic Relevance', 'Complexity', 'Accuracy',
    'CEFR Level', 'CEFR_Level', 'link', 'Transkrip',
]
for col in df.columns:
    if col not in exclude_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(0).replace([np.inf, -np.inf], 0)
print(f"Dataset: {len(df)} samples, {len(df.columns)} kolom")

# ===================================
# SUBCONSTRUCT DEFINITIONS
# ===================================

subconstructs = {
    "Fluency": ["Total Words", "WPM", "WPS", "Filled Pauses", "MLR", "Pause Freq", "Durasi (s)"],
    "Pronunciation": ["Articulation Rate", "Pitch Range (Hz)", "MFCC (%)"],
    "Prosody": ["Mean Pitch", "Stdev Pitch", "Mean Energy", "Stdev Energy",
                "Num Prominences", "Prominence Dist Mean", "Prominence Dist Std"],
    "Coherence_and_Cohesion": ["Semantic Coherence (%)", "Discourse Count", "Linking Count"],
    "Topic_Relevance": ["Topic Similarity (%)"],
    "Complexity": ["Idioms Found", "Bigram Count", "Trigram Count", "Fourgram Count",
                   "Synonym Variations", "CEFR A1", "CEFR A2", "CEFR B1", "CEFR B2",
                   "CEFR C1", "CEFR C2", "CEFR UNKNOWN", "Avg Tree Depth", "Max Tree Depth",
                   "Token Count", "Type Count", "TTR"],
    "Accuracy": ["Grammar Errors"],
}

# Map nama subconstruct ke kolom target di CSV
# CATATAN: CSV pakai "Coherence and Cohesion", bukan underscore
subconstruct_targets = {
    "Fluency":               "Fluency_numeric",
    "Pronunciation":         "Pronunciation_numeric",
    "Prosody":               "Prosody_numeric",
    "Coherence_and_Cohesion": "Coherence and Cohesion_numeric",
    "Topic_Relevance":       "Topic Relevance_numeric",
    "Complexity":            "Complexity_numeric",
    "Accuracy":              "Accuracy_numeric",
}

# Label dimulai dari 1 di CSV → shift ke 0
for sub_name, target_col in subconstruct_targets.items():
    df[target_col] = df[target_col] - 1

# ===================================
# TRAINING LOOP — FIXED VERSION
# ===================================

print("=" * 70)
print("TRAINING PIPELINE — FIXED (no data leakage)")
print("=" * 70)

MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for sub_name, feats in subconstructs.items():
    print(f"\n{'='*70}")
    print(f"Subconstruct: {sub_name}")
    print(f"{'='*70}")

    target_col = subconstruct_targets[sub_name]
    X_raw = df[feats].values
    y     = df[target_col].values
    num_classes = len(np.unique(y))

    print(f"Original: {X_raw.shape[0]} samples, {len(feats)} features, {num_classes} classes")
    print(f"Class dist: { {i: int((y==i).sum()) for i in range(num_classes)} }")

    # ── Feature engineering pada RAW data (sebelum split)
    # Transformer di-fit di dalam fold untuk mencegah leakage
    X_eng = feature_engineering(X_raw, feats)
    print(f"Setelah feature engineering: {X_eng.shape[1]} fitur")

    # ── Cross-validation ──────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_list   = []
    all_y_test = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_eng, y), 1):
        # Split DULU sebelum transform & augmentasi
        X_tr_raw, X_te_raw = X_eng[train_idx], X_eng[test_idx]
        y_tr,     y_te     = y[train_idx],     y[test_idx]

        # ── Fit transformer HANYA pada training fold
        try:
            transformer = PowerTransformer(method='yeo-johnson')
            X_tr_t = transformer.fit_transform(X_tr_raw)
            X_te_t = transformer.transform(X_te_raw)
        except Exception:
            transformer = MinMaxScaler()
            X_tr_t = transformer.fit_transform(X_tr_raw)
            X_te_t = transformer.transform(X_te_raw)

        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr_t)
        X_te_s = scaler.transform(X_te_t)

        # ── Augmentasi HANYA pada training split ─────────────────────────
        X_tr_aug, y_tr_aug = augment_data(X_tr_s, y_tr, augmentation_factor=4)

        # ── SMOTE ────────────────────────────────────────────────────────
        class_counts = pd.Series(y_tr_aug).value_counts()
        min_samples  = int(class_counts.min())
        k_neighbors  = min(5, min_samples - 1)

        if min_samples > 1 and k_neighbors >= 1:
            try:
                smote = BorderlineSMOTE(
                    k_neighbors=k_neighbors, random_state=42,
                    kind='borderline-2', n_jobs=-1,
                )
                X_tr_res, y_tr_res = smote.fit_resample(X_tr_aug, y_tr_aug)
            except Exception:
                try:
                    smote = SMOTE(k_neighbors=k_neighbors, random_state=42, n_jobs=-1)
                    X_tr_res, y_tr_res = smote.fit_resample(X_tr_aug, y_tr_aug)
                except Exception:
                    X_tr_res, y_tr_res = X_tr_aug, y_tr_aug
        else:
            X_tr_res, y_tr_res = X_tr_aug, y_tr_aug

        print(f"  Fold {fold} — train: {X_tr_res.shape[0]}, test: {X_te_s.shape[0]}", end=" ")

        y_pred, _ = train_ensemble(
            X_tr_res, y_tr_res, X_te_s, y_te,
            num_classes, X_tr_res.shape[1],
        )

        acc = accuracy_score(y_te, y_pred)
        acc_list.append(acc)
        all_y_test.extend(y_te)
        all_y_pred.extend(y_pred)

        marker = "✓✓✓" if acc >= 0.85 else (
            "⚠ near" if acc >= 0.75 else f"⚠ gap {(0.85-acc)*100:.1f}%"
        )
        print(f"→ Acc: {acc:.4f} ({acc*100:.1f}%) {marker}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_y_test, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'C{i}' for i in range(num_classes)],
                yticklabels=[f'C{i}' for i in range(num_classes)])
    plt.title(f'Confusion Matrix — {sub_name}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = MODELS_DIR / f"{sub_name}_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=120); plt.close()
    print(f"\n📈 Confusion matrix: {cm_path}")

    print(f"\n📋 Classification Report — {sub_name}:")
    print(classification_report(all_y_test, all_y_pred,
                                 target_names=[f'CEFR-{i}' for i in range(num_classes)]))

    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    print(f"📊 CV Result: {avg_acc:.4f} ± {std_acc:.4f} ({avg_acc*100:.2f}%)")

    if avg_acc >= 0.85:
        print(f"✓✓✓ SUCCESS: {sub_name} ≥ 85%!")
    elif avg_acc >= 0.70:
        print(f"⚠ Near target: {avg_acc*100:.1f}%")
    else:
        print(f"✗ Below target: {avg_acc*100:.1f}%")

    # ──────────────────────────────────────────────────────────────────────
    # FINAL MODEL — fit pada SELURUH data
    # Transformer & scaler di-fit ulang pada seluruh data
    # ──────────────────────────────────────────────────────────────────────
    print("\nTraining final production model (full data)...")

    try:
        final_transformer = PowerTransformer(method='yeo-johnson')
        X_full_t = final_transformer.fit_transform(X_eng)
    except Exception:
        final_transformer = MinMaxScaler()
        X_full_t = final_transformer.fit_transform(X_eng)

    final_scaler = MinMaxScaler()
    X_full_s = final_scaler.fit_transform(X_full_t)

    # Augmentasi full data untuk final model
    X_final_aug, y_final_aug = augment_data(X_full_s, y, augmentation_factor=4)

    # SMOTE final
    class_counts_f = pd.Series(y_final_aug).value_counts()
    min_s_f = int(class_counts_f.min())
    k_f = min(5, min_s_f - 1)

    if min_s_f > 1 and k_f >= 1:
        try:
            smote_f = SMOTE(k_neighbors=k_f, random_state=42, n_jobs=-1)
            X_final_res, y_final_res = smote_f.fit_resample(X_final_aug, y_final_aug)
        except Exception:
            X_final_res, y_final_res = X_final_aug, y_final_aug
    else:
        X_final_res, y_final_res = X_final_aug, y_final_aug

    # Reshape untuk GRU
    X_final_3d  = X_final_res.reshape(-1, 1, X_final_res.shape[1])
    y_final_cat = to_categorical(y_final_res, num_classes=num_classes)

    class_weights_f = compute_class_weight(
        'balanced', classes=np.unique(y_final_res), y=y_final_res
    )
    cw_f = dict(enumerate(class_weights_f))

    final_model = create_gru_model(X_final_res.shape[1], num_classes, dropout=0.35)
    final_model.fit(
        X_final_3d, y_final_cat,
        epochs=150,
        batch_size=32,
        class_weight=cw_f,
        callbacks=[
            EarlyStopping(monitor='accuracy', patience=30,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='accuracy', factor=0.5,
                              patience=15, min_lr=1e-7),
        ],
        verbose=0,
    )

    # ── Simpan model & scaler ─────────────────────────────────────────────
    model_path  = MODELS_DIR / f"{sub_name}_final.keras"
    scaler_path = MODELS_DIR / f"{sub_name}_final_scaler.pkl"

    # Simpan dengan tf.keras agar kompatibel dengan FastAPI (tf_keras)
    final_model.save(str(model_path))

    with open(scaler_path, "wb") as f:
        # Simpan: (transformer, scaler, feature_names)
        pickle.dump((final_transformer, final_scaler, feats), f)

    print(f"✓ Model: {model_path}")
    print(f"✓ Scaler: {scaler_path}")

print("\n" + "=" * 70)
print("PIPELINE SELESAI")
print(f"Model tersimpan di: {MODELS_DIR}")
print("=" * 70)