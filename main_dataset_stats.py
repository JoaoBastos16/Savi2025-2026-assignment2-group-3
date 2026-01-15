import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Diretório do dataset (ajuste conforme necessário)
DATASET = "data/scenes_D/train"

# =========================
# Função para Carregar uma Amostra
# =========================
def load_sample():
    """Carrega uma amostra aleatória de 9 imagens e suas anotações (arquivos .json)"""
    files = [f for f in os.listdir(DATASET) if f.endswith(".png")]
    sample = random.sample(files, 9)

    images = []
    boxes = []

    for f in sample:
        img_path = os.path.join(DATASET, f)
        json_path = os.path.join(DATASET, f.replace(".png", ".json"))

        # Carrega a imagem e as anotações (json)
        img = cv2.imread(img_path, 0)  # Lê em modo escala de cinza
        with open(json_path) as j:
            ann = json.load(j)

        images.append(img)
        boxes.append(ann)

    return images, boxes

# =========================
# Função para Visualizar as Imagens
# =========================
def visualize():
    """Exibe uma amostra de imagens com suas caixas de anotações (bounding boxes)"""
    imgs, anns = load_sample()
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # Aumenta o tamanho da figura para facilitar a visualização

    for ax, img, ann in zip(axes.flatten(), imgs, anns):
        ax.imshow(img, cmap="gray")
        for a in ann:
            x, y, w, h = a["bbox"]
            rect = plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)
            ax.add_patch(rect)
        ax.axis("off")

    plt.tight_layout()  # Ajusta o layout para que as imagens não se sobreponham
    plt.show()

# =========================
# Função para Gerar Estatísticas do Dataset
# =========================
def statistics():
    """Gera e exibe estatísticas sobre o dataset"""
    counts = []  # Quantidade de dígitos por imagem
    sizes = []   # Tamanhos (largura) das caixas de delimitação
    labels = []  # Rótulos das classes

    for f in os.listdir(DATASET):
        if f.endswith(".json"):
            json_path = os.path.join(DATASET, f)
            with open(json_path) as j:
                ann = json.load(j)
                counts.append(len(ann))  # Número de dígitos por imagem
                for a in ann:
                    sizes.append(a["bbox"][2])  # Largura da caixa
                    labels.append(a["label"])    # Rótulo do dígito

    # Exibição das estatísticas
    print(f"Distribuição das classes: {dict(Counter(labels))}")
    print(f"Média de dígitos por imagem: {np.mean(counts):.2f}")
    print(f"Tamanho médio das caixas (largura): {np.mean(sizes):.2f}")

    # Histograma: Número de dígitos por imagem
    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=range(0, max(counts) + 1), edgecolor='black')
    plt.title("Distribuição de dígitos por imagem")
    plt.xlabel("Número de dígitos por imagem")
    plt.ylabel("Frequência")
    plt.show()

    # Histograma: Tamanho das caixas (largura)
    plt.figure(figsize=(8, 6))
    plt.hist(sizes, bins=30, edgecolor='black')
    plt.title("Distribuição do tamanho das caixas")
    plt.xlabel("Largura das caixas")
    plt.ylabel("Frequência")
    plt.show()

# =========================
# Execução Principal
# =========================
if __name__ == "__main__":
    visualize()  # Exibe uma amostra de imagens com caixas
    statistics()  # Exibe as estatísticas do dataset
