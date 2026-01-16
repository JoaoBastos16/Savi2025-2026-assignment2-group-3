import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Diretório do dataset (ajuste conforme necessário)
DATASET = "./Trabalho2/savi_datasets/scenes_D/train"

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
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for ax, img, ann in zip(axes.flatten(), imgs, anns):
        ax.imshow(img, cmap="gray")
        for a in ann:
            x, y, w, h = a["bbox"]
            rect = plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)
            ax.add_patch(rect)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# =========================
# Função para Gerar Estatísticas do Dataset
# =========================
def statistics():
    """Gera e exibe estatísticas sobre o dataset"""
    counts = []  # Quantidade de dígitos por imagem (inteiro)
    sizes = []   # Tamanhos (largura) das caixas (inteiro se o bbox for inteiro)
    labels = []  # Rótulos das classes (inteiro)

    for f in os.listdir(DATASET):
        if f.endswith(".json"):
            json_path = os.path.join(DATASET, f)
            with open(json_path) as j:
                ann = json.load(j)

            counts.append(int(len(ann)))

            for a in ann:
                x, y, w, h = a["bbox"]
                sizes.append(int(w))          # largura
                labels.append(int(a["label"])) # classe

    # Exibição das estatísticas
    print(f"Distribuição das classes: {dict(Counter(labels))}")
    print(f"Média de dígitos por imagem: {np.mean(counts):.2f}")
    print(f"Tamanho médio das caixas (largura): {np.mean(sizes):.2f}")

    # =========================
    # Histograma 1: Nº de dígitos por imagem (1 barra por valor)
    # =========================
    max_c = max(counts) if counts else 0
    # bins com bordas em k-0.5 até k+0.5
    bins_counts = np.arange(0, max_c + 2) - 0.5

    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=bins_counts, edgecolor="black", rwidth=0.9)
    plt.title("Distribuição de dígitos por imagem")
    plt.xlabel("Número de dígitos por imagem")
    plt.ylabel("Frequência")
    plt.xticks(range(0, max_c + 1))  # ticks nos inteiros
    plt.tight_layout()
    plt.show()

    # =========================
    # Histograma 2: Distribuição das classes (0-9) (1 barra por valor)
    # =========================
    bins_labels = np.arange(0, 11) - 0.5  # 0..10 -> bordas -0.5..9.5
    plt.figure(figsize=(8, 6))
    plt.hist(labels, bins=bins_labels, edgecolor="black", rwidth=0.9)
    plt.title("Distribuição das classes (labels)")
    plt.xlabel("Classe do dígito")
    plt.ylabel("Frequência")
    plt.xticks(range(0, 10))
    plt.tight_layout()
    plt.show()

    # =========================
    # Histograma 3: Largura das caixas
    # Se queres MESMO 1 barra por valor (discreto), usa bins assim.
    # Atenção: se houver muitos valores diferentes, pode ficar feio/ilegivél.
    # =========================
    if sizes:
        min_w, max_w = min(sizes), max(sizes)
        bins_sizes = np.arange(min_w, max_w + 2) - 0.5

        plt.figure(figsize=(8, 6))
        plt.hist(sizes, bins=bins_sizes, edgecolor="black", rwidth=0.9)
        plt.title("Distribuição do tamanho das caixas (largura)")
        plt.xlabel("Largura das caixas (px)")
        plt.ylabel("Frequência")

        # Se o range for grande, não metas ticks em todos (fica poluído)
        if (max_w - min_w) <= 40:
            plt.xticks(range(min_w, max_w + 1))
        plt.tight_layout()
        plt.show()

# =========================
# Execução Principal
# =========================
if __name__ == "__main__":
    visualize()    # Exibe uma amostra de imagens com caixas
    statistics()   # Exibe as estatísticas do dataset
    