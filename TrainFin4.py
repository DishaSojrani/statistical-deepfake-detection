# === Deepfake Image Detection using Machine Learning ===

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats
import seaborn as sns
import os
import cv2

# === Project Title ===
PROJECT_TITLE = "Deepfake Image Detection using Machine Learning"

# === Create results folder ===
os.makedirs("results", exist_ok=True)

# === Helper function to save plots ===
def save_plot(fig, filename, subtitle=""):
    if subtitle:
        fig.suptitle(f"{PROJECT_TITLE}\n{subtitle}", fontsize=14, fontweight="bold")
    else:
        fig.suptitle(PROJECT_TITLE, fontsize=14, fontweight="bold")

    safe_title = PROJECT_TITLE.replace(" ", "_").lower()
    filename = f"{safe_title}_{filename}"
    fig.savefig(os.path.join("results", filename), bbox_inches="tight")
    plt.close(fig)

# === Set Paths ===
train_dir = 'dataset/Train'
val_dir = 'dataset/Val'
img_size = (150, 150)
batch_size = 32

# === Data Generators ===
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# === Build CNN Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')   # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === Train the Model ===
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

model.save("Deepfake_Detection.keras")

# === Accuracy and Loss Curves ===
fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.legend()
save_plot(fig, "accuracy_loss.png", subtitle="Training and Validation Curves")

# === Predictions ===
Y_pred = model.predict(val_data)
y_pred = (Y_pred > 0.5).astype("int32").ravel()
y_true = val_data.classes
class_names = list(val_data.class_indices.keys())

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
fig = plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
save_plot(fig, "confusion_matrix.png", subtitle="Model Evaluation")

# === Classification Report ===
report_text = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n")
print(report_text)

safe_title = PROJECT_TITLE.replace(" ", "_").lower()
with open(os.path.join("results", f"{safe_title}_classification_report.txt"), "w") as f:
    f.write(PROJECT_TITLE + "\n\n")
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report_text)

# === ROC Curve ===
fig = plt.figure(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_true, Y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
save_plot(fig, "roc_curve.png", subtitle="Receiver Operating Characteristic")

# === Precision-Recall Curve ===
fig = plt.figure(figsize=(7, 6))
precision, recall, _ = precision_recall_curve(y_true, Y_pred)
ap = average_precision_score(y_true, Y_pred)
plt.plot(recall, precision, label=f"AP = {ap:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
save_plot(fig, "precision_recall.png", subtitle="Precision vs Recall")

# ============================================================
# === 1. DESCRIPTIVE STATISTICS on Pixel Features ============
# ============================================================

def get_pixel_data(folder, num_images=50):
    pixel_values = []
    for root, _, files in os.walk(folder):
        for file in files[:num_images]:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(root, file))
                if img is not None:
                    img = cv2.resize(img, img_size)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    pixel_values.extend(gray.flatten())
    return np.array(pixel_values)

print("\n=== Performing Descriptive Statistics ===")
real_pixels = get_pixel_data(os.path.join(val_dir, class_names[0]))
fake_pixels = get_pixel_data(os.path.join(val_dir, class_names[1]))

def describe_pixels(data, label):
    mean_val = np.mean(data)
    var_val = np.var(data)
    std_val = np.std(data)
    skew_val = stats.skew(data)
    kurt_val = stats.kurtosis(data)
    print(f"{label} -> Mean:{mean_val:.2f}, Var:{var_val:.2f}, Std:{std_val:.2f}, Skew:{skew_val:.2f}, Kurt:{kurt_val:.2f}")
    return mean_val, var_val, std_val, skew_val, kurt_val

describe_pixels(real_pixels, "Real Images")
describe_pixels(fake_pixels, "Fake Images")

# === Histogram Comparison ===
fig = plt.figure(figsize=(8, 6))
plt.hist(real_pixels, bins=50, alpha=0.6, label='Real', color='blue', density=True)
plt.hist(fake_pixels, bins=50, alpha=0.6, label='Fake', color='red', density=True)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Pixel Intensity Distribution")
save_plot(fig, "pixel_histogram.png", subtitle="Descriptive Statistics")

# ============================================================
# === 2. STATISTICAL HYPOTHESIS TESTING =======================
# ============================================================

print("\n=== Performing Statistical Hypothesis Testing ===")

# Chi-square Test
bins = np.linspace(0, 255, 30)
real_hist, _ = np.histogram(real_pixels, bins=bins)
fake_hist, _ = np.histogram(fake_pixels, bins=bins)

# --- Normalize histograms to have the same total count ---
real_hist = real_hist.astype(float)
fake_hist = fake_hist.astype(float)

real_hist = real_hist / np.sum(real_hist)
fake_hist = fake_hist / np.sum(fake_hist)

# Add a small epsilon to avoid zeros
real_hist += 1e-8
fake_hist += 1e-8

# --- Perform Chi-square test ---
chi2, p_chi = stats.chisquare(f_obs=real_hist, f_exp=fake_hist)
print(f"Chi-square Test: Chi2={chi2:.2f}, p-value={p_chi:.4f}")

# Kolmogorov–Smirnov Test
ks_stat, p_ks = stats.ks_2samp(real_pixels, fake_pixels)
print(f"KS Test: Statistic={ks_stat:.2f}, p-value={p_ks:.4f}")

# T-Test (or Mann-Whitney if non-normal)
t_stat, p_t = stats.ttest_ind(real_pixels, fake_pixels, equal_var=False)
print(f"T-Test: Statistic={t_stat:.2f}, p-value={p_t:.4f}")

# Mann-Whitney U Test (non-parametric)
u_stat, p_u = stats.mannwhitneyu(real_pixels, fake_pixels)
print(f"Mann-Whitney U: U={u_stat:.2f}, p-value={p_u:.4f}")

# === Save Hypothesis Results ===
with open(os.path.join("results", f"{safe_title}_stats_results.txt"), "w") as f:
    f.write("=== Descriptive Statistics ===\n")
    f.write(f"Real Mean={np.mean(real_pixels):.2f}, Var={np.var(real_pixels):.2f}\n")
    f.write(f"Fake Mean={np.mean(fake_pixels):.2f}, Var={np.var(fake_pixels):.2f}\n\n")
    f.write("=== Hypothesis Tests ===\n")
    f.write(f"Chi-square: {chi2:.2f}, p={p_chi:.4f}\n")
    f.write(f"KS Test: {ks_stat:.2f}, p={p_ks:.4f}\n")
    f.write(f"T-Test: {t_stat:.2f}, p={p_t:.4f}\n")
    f.write(f"Mann-Whitney U: {u_stat:.2f}, p={p_u:.4f}\n")

# === Boxplot for Comparison ===
fig = plt.figure(figsize=(7, 6))
plt.boxplot([real_pixels, fake_pixels], labels=['Real', 'Fake'])
plt.title("Pixel Distribution Comparison (Boxplot)")
plt.ylabel("Pixel Intensity")
save_plot(fig, "boxplot.png", subtitle="Real vs Fake Pixel Intensity")

# === Violin Plot ===
fig = plt.figure(figsize=(8, 6))
sns.violinplot(data=[real_pixels, fake_pixels])
plt.xticks([0, 1], ['Real', 'Fake'])
plt.title("Pixel Intensity Distribution (Violin Plot)")
save_plot(fig, "violinplot.png", subtitle="Distribution Shape Comparison")

print("✅ All statistical graphs and test results saved in 'results/' folder.")

