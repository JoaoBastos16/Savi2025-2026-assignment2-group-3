#!/usr/bin/env python3
# Trabalho 2 — Tarefa 4
# main_improved_detection.py
#
# Abordagem "leve" (sem sliding window):
# 1) Treina um classificador CNN no MNIST original (60k/10k via torchvision)
# 2) Para cada imagem das scenes (T2), gera "region proposals" por segmentação (componentes conectados)
# 3) Classifica cada proposta com a CNN do MNIST
# 4) Filtra por confiança + NMS e desenha bounding boxes
#
# VS Code Run: configs no topo, sem argparse
# Output fixo: Trabalho2/Outputs/scenes_D

import os
import json
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from tqdm import tqdm

# OpenCV é muito útil aqui (threshold + connected components)
# Se não tiveres: pip install opencv-python
import cv2


# ============================================================
# CONFIGURACAO (MUDA AQUI)
# ============================================================

# ---- Treino MNIST ----
MNIST_ROOT = "./data_mnist"
DO_TRAIN = False
EPOCHS = 6
BATCH_SIZE = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# Guardar modelo
OUTPUT_DIR = "Trabalho2/Outputs/scenes_D"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_cnn_best.pt")

# ---- Avaliacao nas scenes (T2) ----
IMAGES_SOURCE = "./Trabalho2/savi_datasets/scenes_D/train"  # muda facilmente
MAX_IMAGES_EVAL = 0  # 0 => todas, ou por ex. 50 para testes rapidos

SAVE_DRAWINGS = True
SAVE_JSON = True

# Threshold de aceitação do classificador
CONF_THRESH = 0.7

# NMS final (para remover duplicados)
NMS_IOU = 0.25

# ---- Segmentacao / Proposals ----
# Filtros para componentes (em pixels da imagem)
MIN_AREA = 40
MAX_AREA = 5000
MIN_SIDE = 8
MAX_SIDE = 80
ASPECT_MIN = 0.25
ASPECT_MAX = 4.0

# Padding extra em torno da bbox (em pixels)
BBOX_PAD = 2


# ============================================================
# Modelo CNN (treinado em MNIST)
# ============================================================

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Utils: NMS e desenho
# ============================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return float(inter / union)

def nms(dets: List[Dict], iou_thr: float) -> List[Dict]:
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou_xywh(best["bbox"], d["bbox"]) < iou_thr]
    return kept

def draw_detections(img_l: Image.Image, dets: List[Dict]) -> Image.Image:
    out = img_l.convert("RGB")
    draw = ImageDraw.Draw(out)

    for d in dets:
        x, y, w, h = d["bbox"]
        lbl = int(d["label"])
        sc = float(d["score"])

        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=2)
        ty = max(0, y - 14)
        # numero verde, certeza vermelha (sem fundo)
        draw.text((x, ty), str(lbl), fill=(0, 255, 0))
        draw.text((x + 12, ty), f"{sc:.2f}", fill=(255, 0, 0))

    return out


# ============================================================
# Treino MNIST
# ============================================================

def train_mnist(model: nn.Module):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Normalização típica MNIST
    tf_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=tf_train)
    test_ds = datasets.MNIST(root=MNIST_ROOT, train=False, download=True, transform=tf_test)

    # Split treino/val
    val_size = 5000
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_val = float("inf")

    print(f"[MNIST] device={DEVICE} epochs={EPOCHS} batch={BATCH_SIZE} lr={LR}")
    print(f"[MNIST] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"MNIST Epoch {ep}/{EPOCHS} [train]", leave=False):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_loss = float(np.mean(losses)) if losses else 0.0
        train_acc = correct / max(1, total)

        # val loss
        model.eval()
        vlosses = []
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"MNIST Epoch {ep}/{EPOCHS} [val]", leave=False):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                loss = crit(logits, y)
                vlosses.append(float(loss.item()))
                pred = logits.argmax(dim=1)
                vcorrect += int((pred == y).sum().item())
                vtotal += int(y.numel())

        val_loss = float(np.mean(vlosses)) if vlosses else 0.0
        val_acc = vcorrect / max(1, vtotal)

        print(f"[MNIST E{ep:02d}] train_loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} acc={val_acc*100:.2f}%")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> saved best: {MODEL_PATH}")

    # test acc final (opcional)
    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    tcorrect = 0
    ttotal = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            tcorrect += int((pred == y).sum().item())
            ttotal += int(y.numel())
    print(f"[MNIST] test_acc={tcorrect/max(1,ttotal)*100:.2f}%")
    print("[MNIST] done")


# ============================================================
# Proposals por segmentacao (componentes conectados)
# ============================================================

def generate_proposals(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    gray: uint8 [H,W] (0..255)
    returns list of bboxes (x,y,w,h) candidatas
    """
    H, W = gray.shape

    # blur leve para reduzir ruido
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # decidir se precisamos inverter:
    # queremos foreground (dígitos) como branco (255) para componentes conectados
    if bw.mean() > 127:
        bw = cv2.bitwise_not(bw)

    # morfologia (tira pequenos buracos)
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    proposals = []
    for i in range(1, num_labels):  # 0 é background
        x, y, w, h, area = stats[i]

        if area < MIN_AREA or area > MAX_AREA:
            continue
        if w < MIN_SIDE or h < MIN_SIDE or w > MAX_SIDE or h > MAX_SIDE:
            continue

        aspect = w / max(1, h)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        # padding + clamp
        x0 = max(0, x - BBOX_PAD)
        y0 = max(0, y - BBOX_PAD)
        x1 = min(W, x + w + BBOX_PAD)
        y1 = min(H, y + h + BBOX_PAD)

        proposals.append((x0, y0, x1 - x0, y1 - y0))

    return proposals


# ============================================================
# Classificar proposals com CNN MNIST
# ============================================================

def classify_proposals(model: nn.Module, gray: np.ndarray, props: List[Tuple[int, int, int, int]]) -> List[Dict]:
    """
    returns dets: [{label, score, bbox}]
    """
    if not props:
        return []

    # mesma normalização do MNIST
    # (vamos construir tensores manualmente)
    mean, std = 0.1307, 0.3081

    crops = []
    metas = []

    H, W = gray.shape
    for (x, y, w, h) in props:
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        x = max(0, x); y = max(0, y)
        if x2 <= x or y2 <= y:
            continue

        patch = gray[y:y2, x:x2]

        # tornar o dígito "tipo MNIST": dígito claro em fundo escuro costuma funcionar,
        # mas o MNIST é dígito claro em fundo escuro (aprox). Vamos garantir contraste:
        # Se o patch for maioritariamente escuro, invert não; se maioritariamente claro, invert.
        if patch.mean() > 127:
            patch = 255 - patch

        patch28 = cv2.resize(patch, (28, 28), interpolation=cv2.INTER_AREA)

        t = torch.from_numpy(patch28).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,28,28]
        t = (t - mean) / std
        crops.append(t.squeeze(0))  # [1,28,28]
        metas.append((x, y, w, h))

    if not crops:
        return []

    xbat = torch.stack(crops, dim=0).to(DEVICE)  # [N,1,28,28]

    with torch.no_grad():
        logits = model(xbat)
        prob = torch.softmax(logits, dim=1)  # [N,10]
        scores, labels = torch.max(prob, dim=1)

    dets = []
    for (x, y, w, h), s, lbl in zip(metas, scores.cpu().numpy(), labels.cpu().numpy()):
        s = float(s)
        if s < CONF_THRESH:
            continue
        dets.append({"label": int(lbl), "score": s, "bbox": [int(x), int(y), int(w), int(h)]})

    dets = nms(dets, NMS_IOU)
    return dets


# ============================================================
# Avaliar imagens scenes_D
# ============================================================

@torch.no_grad()
def evaluate_scenes(model: nn.Module):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    out_img_dir = os.path.join(OUTPUT_DIR, "images_t4")
    out_det_dir = os.path.join(OUTPUT_DIR, "detections_t4")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_det_dir, exist_ok=True)

    pngs = sorted([p for p in os.listdir(IMAGES_SOURCE) if p.lower().endswith(".png")])
    if not pngs:
        raise FileNotFoundError(f"Não encontrei .png em: {IMAGES_SOURCE}")

    if MAX_IMAGES_EVAL and MAX_IMAGES_EVAL > 0:
        pngs = pngs[:MAX_IMAGES_EVAL]

    print(f"[T4] source={IMAGES_SOURCE}")
    print(f"[T4] n_images={len(pngs)} conf_thresh={CONF_THRESH} nms_iou={NMS_IOU}")
    print(f"[T4] proposal filters: area[{MIN_AREA},{MAX_AREA}] side[{MIN_SIDE},{MAX_SIDE}] aspect[{ASPECT_MIN},{ASPECT_MAX}]")

    times = []
    model.eval()

    for fname in tqdm(pngs, desc="T4 Evaluate", unit="img"):
        path = os.path.join(IMAGES_SOURCE, fname)
        base = os.path.splitext(fname)[0]

        img_l = Image.open(path).convert("L")
        gray = np.array(img_l, dtype=np.uint8)

        t0 = time.time()
        props = generate_proposals(gray)
        dets = classify_proposals(model, gray, props)
        times.append(time.time() - t0)

        if SAVE_JSON:
            out_json = os.path.join(out_det_dir, base + "_detections.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(dets, f, indent=2, ensure_ascii=False)

        if SAVE_DRAWINGS:
            out_img = draw_detections(img_l, dets)
            out_png = os.path.join(out_img_dir, base + "_detections.png")
            out_img.save(out_png)

    avg = float(np.mean(times)) if times else 0.0
    print(f"[T4] avg time per image (proposals+classify+nms): {avg*1000:.2f} ms")
    print(f"[T4] outputs: {out_img_dir} | {out_det_dir}")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = MNIST_CNN().to(DEVICE)

    if DO_TRAIN:
        train_mnist(model)
    else:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[INFO] loaded model: {MODEL_PATH}")

    evaluate_scenes(model)


if __name__ == "__main__":
    main()
