# deepfake_compare_cnn_pca_svm_individual.py
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# sklearn
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# === Project Title & results folder ===
PROJECT_TITLE = "Deepfake Image Detection - CNN vs PCA vs SVM"
os.makedirs("results", exist_ok=True)

def save_plot(fig, filename, subtitle=""):
    if subtitle:
        fig.suptitle(f"{PROJECT_TITLE}\n{subtitle}", fontsize=12, fontweight="bold")
    else:
        fig.suptitle(PROJECT_TITLE, fontsize=12, fontweight="bold")
    safe_title = PROJECT_TITLE.replace(" ", "_").lower().replace(" ", "_")
    filename = f"{safe_title}_{filename}"
    fig.savefig(os.path.join("results", filename), bbox_inches="tight")
    plt.close(fig)

# === Paths and parameters ===
train_dir = 'dataset/Train'
val_dir = 'dataset/Val'
img_size = (150, 150)
channels = 3
random_seed = 42
np.random.seed(random_seed)

# === Helper: load images ===
def load_dataset_from_dirs(base_dir, img_size=(150,150), max_per_class=None):
    X, y = [], []
    class_names = sorted(next(os.walk(base_dir))[1])
    for label, cls in enumerate(class_names):
        folder = os.path.join(base_dir, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if max_per_class:
            files = files[:max_per_class]
        for file in files:
            img = cv2.imread(os.path.join(folder, file))
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y), class_names

# Load dataset
print("Loading dataset...")
X_train, y_train, class_names = load_dataset_from_dirs(train_dir, img_size)
X_val, y_val, _ = load_dataset_from_dirs(val_dir, img_size)
print(f"Classes: {class_names} | Train: {len(X_train)} | Val: {len(X_val)}")

X_train_scaled = X_train / 255.0
X_val_scaled = X_val / 255.0

# ====================================================
# === 1. CNN MODEL ===================================
# ====================================================
def build_cnn(input_shape=(150,150,3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("\n=== Training CNN ===")
cnn_model = build_cnn((img_size[0], img_size[1], channels))
start = time.time()
history = cnn_model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_val_scaled, y_val), batch_size=32, verbose=2)
cnn_time = time.time() - start

y_pred_prob_cnn = cnn_model.predict(X_val_scaled).ravel()
y_pred_cnn = (y_pred_prob_cnn > 0.5).astype(int)

# === CNN Accuracy/Loss ===
fig = plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend(); plt.title("CNN Accuracy")
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend(); plt.title("CNN Loss")
save_plot(fig, "cnn_training.png", subtitle="CNN Training Performance")

# === CNN Confusion Matrix ===
cm_cnn = confusion_matrix(y_val, y_pred_cnn)
fig = plt.figure(figsize=(6,5))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("CNN Confusion Matrix")
save_plot(fig, "cnn_confusion_matrix.png", subtitle="CNN Evaluation")

# === CNN ROC ===
fpr, tpr, _ = roc_curve(y_val, y_pred_prob_cnn)
roc_auc_cnn = roc_auc_score(y_val, y_pred_prob_cnn)
fig = plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_cnn:.3f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("CNN ROC Curve")
save_plot(fig, "cnn_roc.png", subtitle="CNN ROC")

# ====================================================
# === 2. PCA MODEL ===================================
# ====================================================
print("\n=== Running PCA + Logistic Regression ===")
X_train_flat = X_train_scaled.reshape(len(X_train_scaled), -1)
X_val_flat = X_val_scaled.reshape(len(X_val_scaled), -1)

scaler = StandardScaler()
X_train_flat_scaled = scaler.fit_transform(X_train_flat)
X_val_flat_scaled = scaler.transform(X_val_flat)

# PCA to reduce dimensions
pca = PCA(n_components=50, random_state=random_seed)
X_train_pca = pca.fit_transform(X_train_flat_scaled)
X_val_pca = pca.transform(X_val_flat_scaled)

# Train logistic regression on PCA components
logreg = LogisticRegression(max_iter=500)
start = time.time()
logreg.fit(X_train_pca, y_train)
pca_time = time.time() - start

y_pred_prob_pca = logreg.predict_proba(X_val_pca)[:,1]
y_pred_pca = (y_pred_prob_pca > 0.5).astype(int)

cm_pca = confusion_matrix(y_val, y_pred_pca)
fig = plt.figure(figsize=(6,5))
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("PCA Confusion Matrix")
save_plot(fig, "pca_confusion_matrix.png", subtitle="PCA + Logistic Regression")

fpr, tpr, _ = roc_curve(y_val, y_pred_prob_pca)
roc_auc_pca = roc_auc_score(y_val, y_pred_prob_pca)
fig = plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_pca:.3f}", color='orange')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("PCA ROC Curve")
save_plot(fig, "pca_roc.png", subtitle="PCA ROC")

# ====================================================
# === 3. SVM MODEL ===================================
# ====================================================
print("\n=== Training SVM ===")
svm = SVC(kernel='rbf', probability=True, random_state=random_seed)
start = time.time()
svm.fit(X_train_flat_scaled, y_train)
svm_time = time.time() - start

y_pred_prob_svm = svm.predict_proba(X_val_flat_scaled)[:,1]
y_pred_svm = svm.predict(X_val_flat_scaled)

cm_svm = confusion_matrix(y_val, y_pred_svm)
fig = plt.figure(figsize=(6,5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("SVM Confusion Matrix")
save_plot(fig, "svm_confusion_matrix.png", subtitle="SVM Evaluation")

fpr, tpr, _ = roc_curve(y_val, y_pred_prob_svm)
roc_auc_svm = roc_auc_score(y_val, y_pred_prob_svm)
fig = plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_svm:.3f}", color='green')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("SVM ROC Curve")
save_plot(fig, "svm_roc.png", subtitle="SVM ROC")

# ====================================================
# === Compare All Algorithms =========================
# ====================================================
def metrics_summary(y_true, y_pred, y_prob, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob)
    }

summary = [
    metrics_summary(y_val, y_pred_cnn, y_pred_prob_cnn, "CNN"),
    metrics_summary(y_val, y_pred_pca, y_pred_prob_pca, "PCA + Logistic"),
    metrics_summary(y_val, y_pred_svm, y_pred_prob_svm, "SVM")
]

df_summary = pd.DataFrame(summary)
df_summary["TrainTime (s)"] = [cnn_time, pca_time, svm_time]
print("\n=== Algorithm Performance Comparison ===")
print(df_summary.round(4))
df_summary.to_csv("results/deepfake_algorithm_comparison.csv", index=False)

# === Bar Chart for Comparison ===
fig = plt.figure(figsize=(10,6))
x = np.arange(len(df_summary))
plt.bar(x-0.25, df_summary["Accuracy"], width=0.25, label="Accuracy")
plt.bar(x, df_summary["F1"], width=0.25, label="F1")
plt.bar(x+0.25, df_summary["AUC"], width=0.25, label="AUC")
plt.xticks(x, df_summary["Model"])
plt.ylim(0,1.05)
plt.ylabel("Score")
plt.legend()
save_plot(fig, "algorithm_comparison.png", subtitle="CNN vs PCA vs SVM")

# ====================================================
# === Descriptive Statistics & Hypothesis Testing ===
# ====================================================
print("\n=== Descriptive Statistics on Pixel Intensities ===")
def get_pixel_array(X, max_images=50):
    pixels = []
    for img in X[:max_images]:
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        pixels.extend(gray.flatten())
    return np.array(pixels)

real_pixels = get_pixel_array(X_val[y_val==0])
fake_pixels = get_pixel_array(X_val[y_val==1])

def describe_pixels(data, label):
    print(f"{label}: mean={np.mean(data):.2f}, std={np.std(data):.2f}, skew={stats.skew(data):.2f}, kurtosis={stats.kurtosis(data):.2f}")
describe_pixels(real_pixels, "Real")
describe_pixels(fake_pixels, "Fake")

# Hypothesis tests
chi2, p_chi = stats.chisquare(np.histogram(real_pixels, bins=30)[0], np.histogram(fake_pixels, bins=30)[0])
ks, p_ks = stats.ks_2samp(real_pixels, fake_pixels)
t, p_t = stats.ttest_ind(real_pixels, fake_pixels, equal_var=False)
print(f"Chi2={chi2:.3f}, p={p_chi:.4f} | KS={ks:.3f}, p={p_ks:.4f} | T={t:.3f}, p={p_t:.4f}")

fig = plt.figure(figsize=(8,6))
plt.hist(real_pixels, bins=40, alpha=0.6, label='Real')
plt.hist(fake_pixels, bins=40, alpha=0.6, label='Fake')
plt.xlabel("Pixel Intensity"); plt.ylabel("Count"); plt.legend(); plt.title("Pixel Distribution: Real vs Fake")
save_plot(fig, "pixel_distribution.png", subtitle="Descriptive Pixel Statistics")

print("\n✅ Results saved in 'results/' folder (confusion, ROC, stats, and comparison).")
