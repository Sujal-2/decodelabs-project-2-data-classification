# ============================================================
#   DecodeLabs | Industrial Training Kit | Batch 2026
#   Project 2: Data Classification Using AI
#   Algorithm : K-Nearest Neighbors (KNN)
#   Dataset   : Iris Benchmark (150 samples, 3 classes, 4 features)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score
)

# ── STYLING ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#F0F4FF",
    "axes.facecolor":   "#F0F4FF",
    "font.family":      "DejaVu Sans",
})
BLUE   = "#1A3A6B"
ORANGE = "#E85D04"
COLORS = [BLUE, ORANGE, "#2E86AB"]

# ============================================================
# STEP 1 ▸ LOAD & UNDERSTAND THE DATASET
# ============================================================
print("=" * 60)
print("  DecodeLabs | Project 2: Data Classification Using AI")
print("=" * 60)

iris   = load_iris()
X      = iris.data
y      = iris.target
names  = iris.target_names
feat   = iris.feature_names

df = pd.DataFrame(X, columns=feat)
df["Species"] = [names[i] for i in y]

print("\n[1] Dataset Overview")
print(f"    Samples    : {X.shape[0]}")
print(f"    Features   : {X.shape[1]}  → {feat}")
print(f"    Classes    : {len(names)}  → {list(names)}")
print(f"\n{df.describe().round(2)}\n")

# ============================================================
# STEP 2 ▸ FEATURE SCALING  (Gatekeeper Rule)
# ============================================================
print("[2] Applying StandardScaler  (Mean=0, Variance=1)")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("    ✔ Scaling complete\n")

# ============================================================
# STEP 3 ▸ TRAIN / TEST SPLIT  (80 % / 20 %)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, shuffle=True
)
print("[3] Train-Test Split")
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}\n")

# ============================================================
# STEP 4 ▸ TUNE K  (Elbow Method)
# ============================================================
print("[4] Finding Optimal K via Elbow Method …")
error_rates = []
k_range     = range(1, 26)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

best_k = int(np.argmin(error_rates)) + 1
print(f"    ✔ Best K = {best_k}  (lowest error rate: {min(error_rates):.4f})\n")

# ============================================================
# STEP 5 ▸ TRAIN FINAL MODEL
# ============================================================
print(f"[5] Training KNN Classifier  (K={best_k})")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
print("    ✔ Model trained\n")

# ============================================================
# STEP 6 ▸ PREDICT & EVALUATE
# ============================================================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")
cm  = confusion_matrix(y_test, y_pred)

print("[6] Model Evaluation")
print(f"    Accuracy  : {acc * 100:.2f} %")
print(f"    F1 Score  : {f1:.4f}")
print(f"\n    Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=names))

# ============================================================
# STEP 7 ▸ VISUALISATIONS  (4-panel figure)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "DecodeLabs | Project 2: Data Classification Using AI\n"
    "K-Nearest Neighbors on the Iris Benchmark",
    fontsize=14, fontweight="bold", color=BLUE, y=1.01
)

# --- Panel A: Elbow Curve -----------------------------------
ax = axes[0, 0]
ax.plot(k_range, error_rates, color=BLUE, marker="o",
        markersize=5, linewidth=2, label="Error Rate")
ax.axvline(best_k, color=ORANGE, linestyle="--", linewidth=2,
           label=f"Optimal K = {best_k}")
ax.scatter([best_k], [error_rates[best_k - 1]],
           color=ORANGE, s=120, zorder=5)
ax.set_title("Tuning the Engine: Elbow Curve", fontweight="bold", color=BLUE)
ax.set_xlabel("K Value")
ax.set_ylabel("Error Rate")
ax.legend()
ax.grid(alpha=0.3)

# --- Panel B: Confusion Matrix ------------------------------
ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=names, yticklabels=names,
            linewidths=0.5, ax=ax, cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("Diagnostic Tool: Confusion Matrix", fontweight="bold", color=BLUE)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

# --- Panel C: Petal scatter (2 most discriminative features) -
ax = axes[1, 0]
petal_l_idx = 2   # petal length
petal_w_idx = 3   # petal width
for cls, color, label in zip([0, 1, 2], COLORS, names):
    mask = y == cls
    ax.scatter(X[mask, petal_l_idx], X[mask, petal_w_idx],
               color=color, label=label.capitalize(),
               alpha=0.75, edgecolors="white", linewidth=0.4, s=60)
ax.set_title("Architectural Paradigm: Feature Space\n(Petal Length vs Petal Width)",
             fontweight="bold", color=BLUE)
ax.set_xlabel(feat[petal_l_idx])
ax.set_ylabel(feat[petal_w_idx])
ax.legend(title="Species")
ax.grid(alpha=0.3)

# --- Panel D: Per-class F1 bar chart ------------------------
ax = axes[1, 1]
report = classification_report(y_test, y_pred,
                                target_names=names, output_dict=True)
class_f1 = [report[n]["f1-score"] for n in names]
bars = ax.bar(names, class_f1, color=COLORS, edgecolor="white",
              linewidth=0.8)
for bar, val in zip(bars, class_f1):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom",
            fontweight="bold", color=BLUE)
ax.set_ylim(0, 1.15)
ax.set_title("Strategic Trade-Offs: F1 Score per Class",
             fontweight="bold", color=BLUE)
ax.set_ylabel("F1 Score")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = "project2_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[7] Visualisation saved → {out_path}")

# ============================================================
# STEP 8 ▸ LIVE PREDICTION DEMO
# ============================================================
print("\n[8] Live Prediction Demo")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical Setosa
sample_scaled = scaler.transform(sample)
pred  = model.predict(sample_scaled)[0]
proba = model.predict_proba(sample_scaled)[0]

print(f"    Input  : sepal_length=5.1, sepal_width=3.5, "
      f"petal_length=1.4, petal_width=0.2")
print(f"    Output : {names[pred].upper()}  "
      f"(confidence {proba[pred] * 100:.1f} %)")

print("\n" + "=" * 60)
print("  ✅  Project 2 Complete! Badge Earned 🛡")
print("=" * 60)
