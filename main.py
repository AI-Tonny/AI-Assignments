import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)

plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='orange', alpha=0.7, edgecolor='black')

plt.title('Гістограма нормального розподілу')
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.grid(axis='y', alpha=0.3)

plt.savefig('histogram.png')

plt.show()