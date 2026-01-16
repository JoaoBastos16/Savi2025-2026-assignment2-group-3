import os
import random
import json
import cv2
import numpy as np
from torchvision import datasets
from tqdm import tqdm

# =========================
# Configurações Globais
# =========================
IMAGE_SIZE = 128  # Tamanho da imagem final (128x128)
MIN_SCALE = 22  # Escala mínima
MAX_SCALE = 36  # Escala máxima
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # Novo link
OUTPUT_ROOT = "./Trabalho2/savi_datasets"  # Diretório de saída para o dataset gerado

VERSIONS = {
    "A": dict(min_digits=1, max_digits=1, scale=False),  # 1 dígito fixo
    "B": dict(min_digits=1, max_digits=1, scale=True),   # 1 dígito com escala variável
    "C": dict(min_digits=3, max_digits=5, scale=False),  # Múltiplos dígitos fixos
    "D": dict(min_digits=3, max_digits=5, scale=True),   # Múltiplos dígitos com escala variável
}

# =========================
# Funções Utilitárias
# =========================
def iou(box1, box2):
    """Calcula a Interseção sobre a União (IoU) entre duas caixas"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0

def non_overlapping(box, boxes):
    """Verifica se a nova caixa não sobrepõe as caixas existentes"""
    return all(iou(box, b["bbox"]) == 0 for b in boxes)

# =========================
# Geração de Cena (Imagem com Dígitos)
# =========================
def generate_scene(mnist, config):
    """Gera uma imagem com os dígitos do MNIST"""
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    annotations = []

    num_digits = random.randint(config["min_digits"], config["max_digits"])

    for _ in range(num_digits):
        img, label = random.choice(mnist)
        img = np.array(img)

        size = random.randint(MIN_SCALE, MAX_SCALE) if config["scale"] else 28  # Redimensiona de acordo com a escala
        img = cv2.resize(img, (size, size))

        for _ in range(50):  # Tentativas para evitar sobreposição
            x = random.randint(0, IMAGE_SIZE - size)
            y = random.randint(0, IMAGE_SIZE - size)
            box = (x, y, size, size)

            if non_overlapping(box, annotations):
                canvas[y:y+size, x:x+size] = np.maximum(canvas[y:y+size, x:x+size], img)
                annotations.append({
                    "label": int(label),
                    "bbox": box
                })
                break

    return canvas, annotations

# =========================
# Geração do Dataset Completo
# =========================
def generate_dataset(version, split, n_samples):
    """Gera o dataset de cenas com dígitos"""
    output_dir = f"{OUTPUT_ROOT}/scenes_{version}/{split}"
    os.makedirs(output_dir, exist_ok=True)

    mnist = datasets.MNIST(
        root="./Trabalho2/savi_datasets/raw_mnist",
        train=(split == "train"),
        download=True
    )

    for i in tqdm(range(n_samples), desc=f"Gerando {split} {version}"):
        img, ann = generate_scene(mnist, VERSIONS[version])

        # Salva a imagem gerada
        cv2.imwrite(f"{output_dir}/{i:06d}.png", img)

        # Salva as anotações em formato JSON
        with open(f"{output_dir}/{i:06d}.json", "w") as f:
            json.dump(ann, f)

# =========================
# Execução
# =========================
if __name__ == "__main__":
    for v in VERSIONS:
        generate_dataset(v, "train", 60000)  # Gera o conjunto de treino
        generate_dataset(v, "test", 10000)   # Gera o conjunto de teste
