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
    img = Image.open(path).convert("L")  # Načtení v odstínech šedi
    img = img.resize(size)  # Změna velikosti
    img = np.array(img)
    img = np.where(img > 128, 1, -1)  # Převod na binární hodnoty
    return img


# Funkce pro trénování Hopfieldovy sítě
def train_hopfield(patterns):
    size = patterns.shape[1] * patterns.shape[2]
    W = np.zeros((size, size))

    for pattern in patterns:
        p = pattern.flatten()
        W += np.outer(p, p)

    np.fill_diagonal(W, 0)  # Nulování diagonály
    W /= len(patterns)
    return W


# Funkce pro asynchronní iterativní vybavování
def recall(W, pattern, steps=100, noise=0.2):
    noisy_pattern = pattern.copy()

    # Přidání šumu
    mask = np.random.rand(*noisy_pattern.shape) < noise
    noisy_pattern[mask] *= -1

    show_image(noisy_pattern, "Poškozený obrázek")

    # Iterativní obnova
    size = noisy_pattern.size
    recalled = noisy_pattern.flatten()

    for step in range(steps):
        idx = np.random.permutation(size)  # Náhodné pořadí aktualizace
        for i in idx:
            recalled[i] = np.sign(W[i] @ recalled)
        recalled = np.where(recalled == 0, 1, recalled)

        # Zobrazení stavu po určitých iteracích
        if step in [1, 5, 20, 50, 99]:
            show_image(recalled.reshape(32, 32), f"Iterace {step + 1}")

    return recalled.reshape(32, 32)


# Cesty k obrázkům
image_paths = [
    "smile.png",
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

# Trénování Hopfieldovy sítě
W = train_hopfield(images)

# Vybavení z poškozeného vzoru (testujeme na prvním obrázku)
restored_image = recall(W, images[0])
show_image(restored_image, "Obnovený obrázek")
