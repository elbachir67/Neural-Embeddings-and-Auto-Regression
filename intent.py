import matplotlib.pyplot as plt
import numpy as np

# Intent-Aware Experiment Data
translation_pairs = [
    {"name": "Translation Pair 1", "forward": 0.9048, "backward": 0.9680, "quality": 0.9364},
    {"name": "Translation Pair 2", "forward": 0.9048, "backward": 0.9680, "quality": 0.9364},
    {"name": "Translation Pair 3", "forward": 0.9048, "backward": 0.9680, "quality": 0.9364}
]

revision_pairs = [
    {"name": "Revision Pair 1", "forward": 0.9411, "backward": 0.9578, "quality": 0.9453},
    {"name": "Revision Pair 2", "forward": 0.9385, "backward": 0.8938, "quality": 0.9274},
    {"name": "Revision Pair 3", "forward": 0.9406, "backward": 0.9584, "quality": 0.9450}
]

# Create figure with appropriate size for a publication
plt.figure(figsize=(10, 6), dpi=300)

# Set width of bars
barWidth = 0.25

# Set positions of the bars on X axis
r1 = np.arange(len(translation_pairs) + len(revision_pairs))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create bars - first group for forward validation
forward_vals = [pair["forward"] for pair in translation_pairs + revision_pairs]
backward_vals = [pair["backward"] for pair in translation_pairs + revision_pairs]
quality_vals = [pair["quality"] for pair in translation_pairs + revision_pairs]

# Create bars
plt.bar(r1, forward_vals, width=barWidth, color='#3274A1', edgecolor='grey', label='Forward Validation')
plt.bar(r2, backward_vals, width=barWidth, color='#E1812C', edgecolor='grey', label='Backward Validation')
plt.bar(r3, quality_vals, width=barWidth, color='#3A923A', edgecolor='grey', label='Transformation Quality')

# Add a vertical line to separate translation from revision pairs
plt.axvline(x=2.5 + barWidth, color='black', linestyle='--', alpha=0.5)

# Add text labels to indicate sections
plt.text(1.0, 0.83, 'Translation Intent', fontsize=12, fontweight='bold')
plt.text(4.0, 0.83, 'Revision Intent', fontsize=12, fontweight='bold')

# Add labels and title
plt.xlabel('Transformation Pairs', fontweight='bold', fontsize=12)
plt.ylabel('Score', fontweight='bold', fontsize=12)
plt.title('Intent-Aware Experiment Results', fontweight='bold', fontsize=14)

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(translation_pairs) + len(revision_pairs))], 
          [p["name"].replace("Pair ", "P") for p in translation_pairs + revision_pairs], 
          rotation=45)

# Create legend & Show graphic
plt.legend(loc='lower right')
plt.ylim(0.8, 1.0)  # Set y-axis limits to focus on the relevant range
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a tight layout to make sure everything fits well
plt.tight_layout()

# Save the figure
plt.savefig('intent_experiment.png')
plt.close()

print("Intent experiment chart created successfully!")