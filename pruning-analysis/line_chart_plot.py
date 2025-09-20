import matplotlib.pyplot as plt

# Example data
x = [0, 1, 3,  5]  # common x-axis values
ta = [100, 78.4, 42.9, 14.3]  # first line
lib_sr_ft = [80.2, 77.4, 76.8, 75.4]   # second line
lib_sr = [80.2, 64.0, 5, 0]   # third line

# Plot multiple lines
plt.plot(x, ta, label="Action Token Acc", marker="o")
plt.plot(x, lib_sr_ft, label="LIBERO SR (Finetuned)", marker="s")
plt.plot(x, lib_sr, label="LIBERO SR", marker="^")

# Add labels and title
plt.xlabel("Number of Pruned Blocks")
plt.ylabel("%")
plt.title("Pruning Effect on Model Quality")
plt.legend()
plt.grid()

# Show plot
plt.savefig("/home/apilaka/edgevla/openvla/pruning-analysis/pruning_trends.png")
