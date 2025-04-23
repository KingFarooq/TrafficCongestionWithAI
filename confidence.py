import matplotlib.pyplot as plt

epochs = list(range(1, 21))
avg_accuracy = [0.1045, 0.1569, 0.1566, 0.1675, 0.1099, 0.1080, 0.1249, 0.0701, 0.2307, 0.1385,
                0.1399, 0.2128, 0.1768, 0.1366, 0.1534, 0.1386, 0.1693, 0.1692, 0.1387, 0.1372]

plt.figure(figsize=(5, 2))  # Small graph size
plt.bar(epochs, avg_accuracy, color="blue", width=0.4)

plt.xlabel("Epochs", fontsize=8)
plt.ylabel("Avg Acc", fontsize=8)
plt.title("Compact Accuracy Bar Chart", fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5)

plt.show()
