import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#  1. LOAD SAVED METRICS
with open("cnn_metrics.json", "r") as f:
    cnn = json.load(f)

with open("transfer_learning_metrics.json", "r") as f:
    tl_models = json.load(f)

# Combine all models
all_models = [cnn] + tl_models

model_names = [m["model"]    for m in all_models]
accuracies  = [m["accuracy"]  for m in all_models]
precisions  = [m["precision"] for m in all_models]
recalls     = [m["recall"]    for m in all_models]
f1_scores   = [m["f1_score"]  for m in all_models]


#  2. PRINT COMPARISON TABLE
print("\n" + "=" * 70)
print(f"  {'MODEL':<20} {'ACCURACY':>10} {'PRECISION':>11} {'RECALL':>8} {'F1-SCORE':>10}")
print("=" * 70)

best_idx = f1_scores.index(max(f1_scores))
for i, m in enumerate(all_models):
    tag = "  BEST" if i == best_idx else ""
    print(f"  {m['model']:<20} {m['accuracy']:>10.4f} {m['precision']:>11.4f} "
          f"{m['recall']:>8.4f} {m['f1_score']:>10.4f}{tag}")

print("=" * 70)
print(f"\n Best Model: {all_models[best_idx]['model']} "
      f"(F1 = {f1_scores[best_idx]:.4f})")


#  3. BAR CHART — ALL METRICS SIDE BY SIDE
x = np.arange(len(model_names))
width = 0.2
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("Model Comparison — All Metrics", fontsize=15, fontweight="bold")

bars1 = ax.bar(x - 1.5 * width, accuracies,  width, label="Accuracy",  color=colors[0])
bars2 = ax.bar(x - 0.5 * width, precisions,  width, label="Precision", color=colors[1])
bars3 = ax.bar(x + 0.5 * width, recalls,     width, label="Recall",    color=colors[2])
bars4 = ax.bar(x + 1.5 * width, f1_scores,   width, label="F1-Score",  color=colors[3])

# Add value labels on top of bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Highlight best model
ax.axvspan(best_idx - 0.45, best_idx + 0.45, alpha=0.08, color="gold")
ax.text(best_idx, 1.08, "BEST", ha="center", fontsize=10,
        color="darkgoldenrod", fontweight="bold")

plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=150)
plt.show()
print("Bar chart saved as 'model_comparison_bar.png'")


#  4. RADAR CHART (Spider Plot)
categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.suptitle("Model Comparison — Radar Chart", fontsize=14, fontweight="bold")

radar_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for i, m in enumerate(all_models):
    values = [m["accuracy"], m["precision"], m["recall"], m["f1_score"]]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2,
            label=m["model"], color=radar_colors[i])
    ax.fill(angles, values, alpha=0.08, color=radar_colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig("model_comparison_radar.png", dpi=150)
plt.show()
print(" Radar chart saved as 'model_comparison_radar.png'")


#  5. F1-SCORE RANKING CHART
sorted_models = sorted(all_models, key=lambda x: x["f1_score"], reverse=True)
sorted_names  = [m["model"]    for m in sorted_models]
sorted_f1     = [m["f1_score"] for m in sorted_models]
bar_colors    = ["gold" if i == 0 else "#6baed6" for i in range(len(sorted_models))]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(sorted_names, sorted_f1, color=bar_colors, edgecolor="gray")
ax.set_xlim(0, 1.1)
ax.set_xlabel("F1-Score", fontsize=12)
ax.set_title("Model Ranking by F1-Score", fontsize=13, fontweight="bold")

for bar, val in zip(bars, sorted_f1):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=11)

ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("model_ranking_f1.png", dpi=150)
plt.show()
print("Ranking chart saved as 'model_ranking_f1.png'")


#  6. WRITTEN SUMMARY REPORT
best_model = all_models[best_idx]

report = f"""

        AERIAL OBJECT CLASSIFICATION — MODEL COMPARISON REPORT

  METRIC SUMMARY
  {'Model':<20} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8}
  {'-'*60}"""

for m in all_models:
    tag = " " if m["model"] == best_model["model"] else ""
    report += f"""
  {m['model']:<20} {m['accuracy']:>10.4f} {m['precision']:>11.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f}{tag}"""

report += f"""


  MODEL ANALYSIS
1. Custom CNN (Accuracy: {all_models[0]['accuracy']:.2%})
   - Built from scratch with 3 Conv blocks
   - High precision (93.75%) but low recall (63.83%)
   - Struggles to correctly detect drones — misclassifies many as birds
   - Unstable validation curve indicates overfitting

2. ResNet50 (Accuracy: {all_models[1]['accuracy']:.2%})
   - Pre-trained deep residual network
   - Balanced precision and recall (~82-85%)
   - Validation loss was unstable during training
   - Decent performance but not best for this dataset

3. MobileNetV2 (Accuracy: {all_models[2]['accuracy']:.2%}) BEST
   - Lightweight, efficient architecture designed for real-world use
   - Near-perfect precision (98.91%) AND recall (96.81%)
   - Extremely stable training — lowest loss, highest accuracy
   - Only 4 total misclassifications on 215 test images
   - RECOMMENDED for Streamlit deployment

4. EfficientNetB0 (Accuracy: {all_models[3]['accuracy']:.2%}) FAILED
   - Model collapsed — predicted almost everything as "bird"
   - Recall of only 8.51% for drones makes it unusable
   - Likely due to incompatibility with current TF/Keras version
   - Not recommended for deployment


  CONCLUSION
  Best Model : {best_model['model']}
  F1-Score   : {best_model['f1_score']:.4f}
  Accuracy   : {best_model['accuracy']:.4f}

  MobileNetV2 is selected for Streamlit deployment.
  Saved as: best_MobileNetV2_finetuned.keras

"""

print(report)

with open("model_comparison_report.txt", "w") as f:
    f.write(report)

print("Full report saved as 'model_comparison_report.txt'")

