import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Funkce pro zobrazení obrázku
def show_image(data, title="", cmap="gray"):
    plt.figure(figsize=(4, 4))
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Načtení obrázku a převod na binární hodnoty
def load_image(path, size=(32, 32)):
    img = Image.open(path).convert("L")  # Načtení v odstínech šedSi
    img = img.resize(size)  # Změna velikosti
    img = np.array(img)
    img = np.where(img > 128, 1, -1)  # Převod na binární hodnoty
    return img


# Storkeyho pravidlo pro učení Hopfieldovy sítě
def train_hopfield_storkey(patterns):
    size = patterns.shape[1] * patterns.shape[2]
    W = np.zeros((size, size))

    for pattern in patterns:
        p = pattern.flatten()
        for i in range(size):
            for j in range(size):
                if i != j:
                    h_ij = np.dot(W[i], p) - W[i, j] * p[j]
                    h_ji = np.dot(W[j], p) - W[j, i] * p[i]
                    W[i, j] += (p[i] * p[j] - p[i] * h_ji - p[j] * h_ij) / (2*size)
    return W


# Výpočet energie Hopfieldovy sítě
def energy(W, state):
    return -0.5 * np.dot(state, np.dot(W, state))


# Funkce pro asynchronní iterativní vybavování s energií
def recall(W, pattern, steps=10, noise=0.8):
    noisy_pattern = pattern.copy()

    # Přidání šumu
    mask = np.random.rand(*noisy_pattern.shape) < noise
    noisy_pattern[mask] *= -1

    show_image(noisy_pattern, "Poškozený obrázek")

    # Iterativní obnova
    size = noisy_pattern.size
    recalled = noisy_pattern.flatten()
    energy_levels = []

    for step in range(steps):
        idx = np.random.permutation(size)  # Náhodné pořadí aktualizace
        for i in idx:
            recalled[i] = np.sign(W[i] @ recalled)
        recalled = np.where(recalled == 0, 1, recalled)

        # Ukládáme energii
        energy_levels.append(energy(W, recalled))

        # Zobrazení stavu po určitých iteracích
        if step in [0, 1, 2]:
            show_image(recalled.reshape(32, 32), f"Iterace {step + 1}")

    # Graf energie během iterací
    plt.figure(figsize=(5, 3))
    plt.plot(energy_levels, marker='o', linestyle='-')
    plt.xlabel("Iterace")
    plt.ylabel("Energie")
    plt.title("Vývoj energie Hopfieldovy sítě")
    plt.show()

    return recalled.reshape(32, 32)


# Cesty k obrázkům
image_paths = [
    "H.png",
    "heart.png",
    "cross.png",
    "arrow.png"
]

# Načtení obrázků
images = np.array([load_image(f"{path}") for path in image_paths])

# Zobrazení původních obrázků
for i, img in enumerate(images):
    show_image(img, f"Původní obrázek {i + 1}")

# Trénování Hopfieldovy sítě pomocí Storkeyho pravidla
W = train_hopfield_storkey(images)

# Vybavení z poškozeného vzoru (testujeme na prvním obrázku)
restored_image = recall(W, images[0])
show_image(restored_image, "Obnovený obrázek")