import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(32)

alive_aura = np.random.normal(0.7, 0.10, 50)
alive_speed = np.random.normal(0.8, 0.10, 50)

dead_aura = np.random.normal(0.2, 0.10, 50)
dead_speed = np.random.normal(0.3, 0.10, 50)

X = np.vstack((
    np.column_stack((dead_aura, dead_speed)),
    np.column_stack((alive_aura, alive_speed))
))
y = np.array([0] * 50 + [1] * 50)

class SurvivalPredictor:
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

model = SurvivalPredictor(n_inputs=2)
model.fit(X, y)


def plot_survival_logic(model, X, y):
    cmap_light = ListedColormap(['#E6E6FA', '#F0E68C'])

    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='black', label='YOU WILL DIE ☠️')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='YOU WILL LIVE 😉')
    plt.title("Survival Prediction")
    plt.xlabel("Aura level (0-1)")
    plt.ylabel("Speed level (0-1)")
    plt.legend()
    plt.show()


plot_survival_logic(model, X, y)

characters_data = np.array([
    [0.8, 0.4],
    [0.5, 0.6],
    [0.3, 0.6],
    [0.6, 0.4]
])

predictions = model.predict(characters_data)
for i, prediction in enumerate(predictions):
    label = "You will survive, BECAUSE of you " if prediction == 1 else "You're gonna die :("
    print(f'Character #{i+1}. {label}')