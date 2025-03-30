import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiments.experiment_01.scripts.Perceptron import Perceptron

iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # binární klasifikace: Setosa vs ostatní

# Normalizuj vstupy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rozděl na trénovací a testovací množinu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Parametry experimentu
n_experiments = 10
learning_rate = 0.1
epochs = 100

all_errors = []
all_accuracies = []
models = []

for i in range(n_experiments):
    p = Perceptron(input_size=X.shape[1], learning_rate=learning_rate, epochs=epochs)
    errors = p.train(X_train, y_train)
    acc = p.test(X_test, y_test)

    all_errors.append(errors)
    all_accuracies.append(acc)
    models.append((p.weights.copy(), acc))

    # Ulož průběh chyby a váhy
    np.save(f"errors_run_{i}.npy", errors)
    np.save(f"weights_run_{i}.npy", p.weights)

# Boxplot trénovacích chyb (posledních 10 epoch)
plt.figure(figsize=(10, 6))
plt.boxplot([e[-10:] for e in all_errors])
plt.title("Trénovací chyba (posledních 10 epoch) pro 10 běhů")
plt.xlabel("Číslo experimentu")
plt.ylabel("Počet chyb")
plt.grid(True)
plt.savefig("perceptron_boxplot.png")
plt.show()

# Výpis nejlepšího modelu
best_idx = np.argmax(all_accuracies)
best_weights = models[best_idx][0]
best_accuracy = models[best_idx][1]

print(f"Nejlepší model: běh {best_idx}, přesnost: {best_accuracy}")
print("Váhy nejlepšího modelu:", best_weights)