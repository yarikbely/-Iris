import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os


class Neuron:
    """
    Искусственный нейрон с сигмоидной функцией активации
    """

    def __init__(self, n_inputs, learning_rate=0.01):
        """
        Инициализация нейрона
        """
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        """Сигмоидная функция активации"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Прямой проход
        X: (n_samples, n_features)
        возвращает: (n_samples,)
        """
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias
        self.a = self.sigmoid(self.z)
        return self.a

    def backward(self, d_a):
        """
        Обратный проход
        d_a: градиент потерь по выходу (n_samples,)
        """
        # Производная сигмоиды
        d_z = d_a * self.sigmoid(self.z) * (1 - self.sigmoid(self.z))

        # Градиенты
        d_weights = np.dot(self.X.T, d_z) / len(d_z)
        d_bias = np.mean(d_z)

        # Обновление параметров
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        # Градиент для предыдущего слоя
        d_input = np.outer(d_z, self.weights)

        return d_input

    def predict(self, X, threshold=0.5):
        """Предсказание класса"""
        output = self.forward(X)
        return (output >= threshold).astype(int)


class Layer:
    """Слой нейронной сети"""

    def __init__(self, n_inputs, n_neurons, learning_rate=0.01):
        self.neurons = [Neuron(n_inputs, learning_rate) for _ in range(n_neurons)]

    def forward(self, X):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(X))
        return np.array(outputs).T

    def backward(self, d_output):
        d_inputs = []
        for i, neuron in enumerate(self.neurons):
            d_input = neuron.backward(d_output[:, i])
            d_inputs.append(d_input)
        return np.sum(d_inputs, axis=0)


class NeuralNetwork:
    """Многослойная нейронная сеть"""

    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], learning_rate))

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, d_output):
        d_current = d_output
        for layer in reversed(self.layers):
            d_current = layer.backward(d_current)

    def train(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """Обучение сети"""
        n_samples = X.shape[0]
        losses = []
        accuracies = []

        for epoch in range(epochs):
            # Перемешивание
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Прямой проход
                output = self.forward(X_batch).flatten()

                # Loss (бинарная кросс-энтропия)
                epsilon = 1e-7
                output = np.clip(output, epsilon, 1 - epsilon)
                loss = -np.mean(y_batch * np.log(output) + (1 - y_batch) * np.log(1 - output))
                epoch_loss += loss

                # Обратный проход
                d_output = -(y_batch - output)
                self.backward(d_output.reshape(-1, 1))

            avg_loss = epoch_loss / (n_samples / batch_size)
            losses.append(avg_loss)

            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)
            accuracies.append(accuracy)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.4f}")

        return losses, accuracies

    def predict(self, X, threshold=0.5):
        output = self.forward(X).flatten()
        return (output >= threshold).astype(int)


def load_and_prepare_iris():
    """Загрузка и подготовка данных Iris"""
    print("Загрузка датасета Iris...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Бинарная классификация: класс 0 vs класс 1
    binary_mask = (y == 0) | (y == 1)
    X = X[binary_mask]
    y = y[binary_mask]

    # Нормализация
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Классы: 0 ({sum(y == 0)}), 1 ({sum(y == 1)})")

    return X_train, X_test, y_train, y_test, scaler


def plot_decision_boundary(model, X, y, title, save_path=None):
    """Отрисовка разделяющей линии"""
    # Используем первые два признака
    X_vis = X[:, :2]

    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    full_grid = np.zeros((grid_points.shape[0], X.shape[1]))
    full_grid[:, :2] = grid_points

    Z = model.predict(full_grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5)

    colors = ['blue' if label == 0 else 'red' for label in y]
    ax.scatter(X_vis[:, 0], X_vis[:, 1], c=colors, s=50, edgecolors='black')

    ax.set_xlabel('Признак 1 (нормализованный)')
    ax.set_ylabel('Признак 2 (нормализованный)')
    ax.set_title(title)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Класс 0'),
                       Patch(facecolor='red', alpha=0.7, label='Класс 1')]
    ax.legend(handles=legend_elements)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_loss_curves(losses, accuracies, title, save_path=None):
    """Отрисовка кривых обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(losses, 'b-')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Функция потерь')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(accuracies, 'r-')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Точность')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Оценка модели"""
    print(f"\n{'=' * 50}")
    print(f"ОЦЕНКА МОДЕЛИ: {model_name}")
    print(f"{'=' * 50}")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f"\nМетрики на обучающей выборке:")
    print(f"  Accuracy: {train_acc:.4f}")

    print(f"\nМетрики на тестовой выборке:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nМатрица ошибок:\n{cm}")

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    """Основная функция"""
    # Создание папок
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №4")
    print("НЕЙРОННАЯ СЕТЬ С НУЛЯ")
    print("=" * 60)

    # Загрузка данных
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_iris()

    # ============================================
    # ЗАДАЧА 1: Один нейрон
    # ============================================
    print("\n" + "=" * 60)
    print("ЗАДАЧА 1: ОДИН НЕЙРОН")
    print("=" * 60)

    single_neuron = Neuron(n_inputs=X_train.shape[1], learning_rate=0.1)

    print("\nОбучение одного нейрона...")
    losses_1 = []
    accuracies_1 = []

    for epoch in range(500):
        # Прямой проход
        output = single_neuron.forward(X_train)

        # Loss
        epsilon = 1e-7
        output_clipped = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.mean(y_train * np.log(output_clipped) +
                        (1 - y_train) * np.log(1 - output_clipped))
        losses_1.append(loss)

        # Обратный проход
        d_output = -(y_train - output_clipped)
        single_neuron.backward(d_output)

        # Точность
        predictions = single_neuron.predict(X_train)
        accuracy = np.mean(predictions == y_train)
        accuracies_1.append(accuracy)

        if epoch % 100 == 0 or epoch == 499:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.4f}")

    results_1 = evaluate_model(single_neuron, X_train, X_test, y_train, y_test, "Один нейрон")

    plot_decision_boundary(
        single_neuron, X_train, y_train,
        "Разделяющая линия: Один нейрон",
        save_path='results/decision_boundary_1neuron.png'
    )

    plot_loss_curves(
        losses_1, accuracies_1,
        "Обучение одного нейрона",
        save_path='results/training_1neuron.png'
    )

    # ============================================
    # ЗАДАЧА 2: Нейронная сеть (2 слоя по 10 нейронов)
    # ============================================
    print("\n" + "=" * 60)
    print("ЗАДАЧА 2: Нейронная сеть (2 слоя x 10 нейронов)")
    print("=" * 60)

    nn = NeuralNetwork(layer_sizes=[X_train.shape[1], 10, 1], learning_rate=0.1)

    losses_2, accuracies_2 = nn.train(X_train, y_train, epochs=500, batch_size=16, verbose=True)

    results_2 = evaluate_model(nn, X_train, X_test, y_train, y_test, "2 слоя x 10 нейронов")

    plot_decision_boundary(
        nn, X_train, y_train,
        "Разделяющая линия: 2 слоя x 10 нейронов",
        save_path='results/decision_boundary_2layers.png'
    )

    plot_loss_curves(
        losses_2, accuracies_2,
        "Обучение нейронной сети (10-10)",
        save_path='results/training_2layers.png'
    )

    # ============================================
    # СРАВНЕНИЕ
    # ============================================
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    print("\n{:<25} {:>15} {:>15}".format("Метрика", "1 нейрон", "2 слоя x 10"))
    print("-" * 55)

    metrics = ['test_accuracy', 'precision', 'recall', 'f1']
    names = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    for metric, name in zip(metrics, names):
        val1 = results_1[metric]
        val2 = results_2[metric]
        diff = val2 - val1
        arrow = "↑" if diff > 0 else "↓"
        print("{:<25} {:>14.4f} {:>14.4f} {:>10.4f} {}".format(
            name, val1, val2, diff, arrow
        ))

    print("\n" + "=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)
    print("""
1. Один нейрон (логистическая регрессия):
   - Создает линейную разделяющую границу
   - Простая модель с 5 параметрами

2. Нейронная сеть (2 слоя x 10 нейронов):
   - Может создавать нелинейные границы
   - 61 параметр для обучения

3. Сравнение:
   - Обе модели успешно классифицируют Iris
   - Нейронная сеть может быть точнее на сложных данных
   - Один нейрон обучается быстрее
    """)

    print("\nРезультаты сохранены в папке results/")


if __name__ == "__main__":
    main()