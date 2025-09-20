import matplotlib.pyplot as plt
import numpy as np

# Number of categories (benchmarks) and models
n_categories = 10
n_models = 4
PRE_FT = True

data = np.array(
    [[82.00, 72.00, 88.0, 92.00, 68.00, 86.00, 94.00, 78.00, 72.00, 70.00], # baseline edgevla
     [66.00, 82.00, 70.00, 88.00, 74.00, 80.00, 86.00, 84.00, 78.00, 66.00],# one block ft
     [76.00, 66.00, 84.00, 82.00, 74.00, 86.00, 90.00, 60.00, 80.00, 70.00],# three block ft
     [84.00, 70.00, 74.00, 96.00, 60.00, 84.00, 90.00, 70.00, 86.00, 40.00]], dtype=np.float32)  # five block ft

data = data if not PRE_FT else np.array(
    [[82.00, 72.00, 88.0, 92.00, 68.00, 86.00, 94.00, 78.00, 72.00, 70.00], # baseline edgevla
     [80.00, 64.00, 54.00, 82.00, 50.00, 74.00, 66.00, 54.00, 72.00, 44.00],# one block ft
     [0, 4.00, 8.00, 0, 0, 0, 0, 0, 0, 0],# three block ft
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32) # five block ft

# Category labels (Benchmarks)
categories = ["bowl between plate and ramekin", "bowl next to ramekin", "bowl from table center", "bowl on cookie box", "bowl in top drawer", "bowl on ramekin", "bowl next to cookie", "bowl on stove", "bowl next to plate", "bowl on wooden cabinet"]
# Model labels
model_labels = ["Baseline EdgeVLA", "EdgeVLA Pruned 1", "EdgeVLA Pruned 3", "EdgeVLA Pruned 5"]
colors = ["blue", "red", "orange", "green"]

# X positions for categories
x = np.arange(n_categories)

# Width of each bar
bar_width = 0.2

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

for i in range(n_models):
    ax.bar(x + i * bar_width, data[i], width=bar_width,
           label=model_labels[i], color=colors[i])

# Formatting
ax.set_xlabel("LIBERO Spatial Task")
ax.set_ylabel("Success Rate")
ax.set_title(f"Model Performance on LIBERO Spatial Tasks {'before' if PRE_FT else 'after' } Finetuning")
ax.set_xticks(x + bar_width * (n_models - 1) / 2)
ax.set_xticklabels(categories, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.savefig(f"/home/apilaka/edgevla/openvla/pruning-analysis/LIBERO_subtask_barchart_{'noft' if PRE_FT else 'ft'}.png")
