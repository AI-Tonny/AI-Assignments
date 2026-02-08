import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------
# 1. Датасет: Побачення чи ігнор?
# ---------------------------
np.random.seed(10)

# Клас 0: Краще залишитись вдома (нудні або... специфічні)
skip_beauty = np.random.normal(0.3, 0.15, 50)
skip_humor = np.random.normal(0.2, 0.1, 50)

# Клас 1: Йти однозначно! (Гарні та смішні)
date_beauty = np.random.normal(0.7, 0.15, 50)
date_humor = np.random.normal(0.8, 0.1, 50)

X = np.vstack((
    np.column_stack((skip_beauty, skip_humor)),
    np.column_stack((date_beauty, date_humor))
))
y = np.array([0] * 50 + [1] * 50)


# ---------------------------
# 2. Тюнінг: Імпульсивний ШІ (lr = 10.0)
# ---------------------------
class DatePicker:
    def __init__(self, n_inputs, learning_rate=0.1, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, 0)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update


model = DatePicker(n_inputs=2)
model.fit(X, y)


# ---------------------------
# 3. Візуалізація
# ---------------------------
def plot_date_logic(model, X, y):
    cmap_light = ListedColormap(['#FFB6C1', '#87CEEB'])

    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='purple', label='Френдзона 💜')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Шанси є ❤️')
    plt.title("ДЕТЕКТОР ПЕРСПЕКТИВНОГО ВЕЧОРА")
    plt.xlabel("Рівень краси (0-1)")
    plt.ylabel("Почуття гумору (0-1)")
    plt.legend()
    plt.show()


plot_date_logic(model, X, y)

# ---------------------------
# 4. Вердикти для випадків
# ---------------------------
new_samples = np.array([
    [0.8, 0.4],
    [0.5, 0.6],
    [0.3, 0.6],
    [0.6, 0.4]
])

predictions = model.predict(new_samples)
for i, prediction in enumerate(predictions):
    label = "Go on a date RIGHT NOW!!!" if prediction == 1 else "It's better to stay home today dude."
    print(f'Guy #{i+1}. {label}')