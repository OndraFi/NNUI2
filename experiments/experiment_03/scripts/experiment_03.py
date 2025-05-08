from keras import Sequential
from keras.src.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

# Načtení dat
iris = load_iris()
X = iris.data
y = iris.target

# Normalizace
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding výstupů
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Rozdělení dat: trénink (60%), validace (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

import numpy as np

def build_model(topology):
    model = Sequential()
    model.add(Dense(topology[0], activation='relu', input_shape=(4,)))
    for units in topology[1:-1]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 třídy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

topologies = [
    [8, 3],
    [16, 8, 3],
    [32, 16, 8, 3],
    [64, 32, 16, 8, 3],
    [128, 64, 32, 16, 8, 3]
]

history_per_topology = {}

for idx, topology in enumerate(topologies):
    acc_histories = []
    for run in range(10):
        model = build_model(topology)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, verbose=0)
        acc_histories.append(history.history['val_accuracy'][-1])  # poslední validační přesnost
    history_per_topology[f'Topology_{idx+1}'] = acc_histories



# Boxplot
plt.boxplot(history_per_topology.values(), labels=history_per_topology.keys())
plt.title('Boxplot validační přesnosti jednotlivých topologií')
plt.ylabel('Validační přesnost')
plt.xlabel('Topologie')
plt.grid(True)
plt.savefig('boxplot_val_accuracy.png')  # uloží obrázek
plt.show()



# Příklad: zjistíme nejlepší topologii
best_topology_name = max(history_per_topology, key=lambda k: np.median(history_per_topology[k]))
best_topology_index = list(history_per_topology.keys()).index(best_topology_name)
best_topology = topologies[best_topology_index]

# Natrénujeme finální model
final_model = build_model(best_topology)
final_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0)

# Testování
test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

final_model.save('best_model.h5')  # uloží celý model i s váhami