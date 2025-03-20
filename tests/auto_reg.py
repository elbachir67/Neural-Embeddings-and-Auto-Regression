import matplotlib.pyplot as plt
import numpy as np

# Auto-Regression Experiment Data
with_auto = {"forward": 0.9424, "backward": 0.9574, "quality": 0.9461}
without_auto = {"forward": 0.8823, "backward": 0.9590, "quality": 0.9206}
improvement = {
    "forward": with_auto["forward"] - without_auto["forward"],
    "backward": with_auto["backward"] - without_auto["backward"],
    "quality": with_auto["quality"] - without_auto["quality"]
}

# Create figure with appropriate size for a publication
plt.figure(figsize=(10, 6), dpi=300)

# Set width of bars
barWidth = 0.3

# Set positions of the bars on X axis
r1 = np.arange(3)
r2 = [x + barWidth for x in r1]

# Labels for x-axis
metrics = ['Forward Validation', 'Backward Validation', 'Transformation Quality']

# Create bars
plt.bar(r1, [with_auto["forward"], with_auto["backward"], with_auto["quality"]], 
        width=barWidth, color='#3274A1', edgecolor='grey', 
        label='With Auto-Regression')
plt.bar(r2, [without_auto["forward"], without_auto["backward"], without_auto["quality"]], 
        width=barWidth, color='#E1812C', edgecolor='grey', 
        label='Without Auto-Regression')

# Add text for improvement percentage on the quality metric
plt.text(r1[2]+barWidth/2, 0.93, 
         f"+{improvement['quality']*100:.1f}%", 
         ha='center', va='bottom', fontweight='bold', color='green')

# Add labels and title
plt.xlabel('Metrics', fontweight='bold', fontsize=12)
plt.ylabel('Score', fontweight='bold', fontsize=12)
plt.title('Auto-Regression Experiment Results', fontweight='bold', fontsize=14)

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth/2 for r in r1], metrics)

# Create legend & Show graphic
plt.legend(loc='lower right')
plt.ylim(0.85, 1.0)  # Set y-axis limits to focus on the relevant range
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a tight layout to make sure everything fits well
plt.tight_layout()

# Save the figure
plt.savefig('auto_regression_experiment.png')
plt.close()

print("Auto-regression experiment chart created successfully!")