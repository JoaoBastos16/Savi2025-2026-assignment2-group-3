#!/usr/bin/env python3
# Trabalho 2 — Tarefa 3
# Sliding Window detector usando a CNN da Tarefa 1
# VS Code Run: configs no topo, sem argparse
#
# Pós-processamento pedido:
# - Se houver caixas sobrepostas (IoU >= OVERLAP_IOU),
#   mantém a que tem mais "branco" (média de intensidade maior).

import os
import json
import glob
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURACAO (MUDA AQUI)
# ============================================================

IMAGES_SOURCE = "./Trabalho2/savi_datasets/scenes_D/train"
MODEL_PATH = "runs/t1/best_model.pt"
OUTPUT_DIR = "Trabalho2/Outputs/scenes_D"

# Sliding window
STRIDE = 8
WIN_SIZES = [20, 24, 28, 32, 36]
BATCH_SIZE = 256

# Critérios para aceitar uma deteção (antes do overlap-filter)
CONF_THRESH = 0.95
MARGIN_THRESH = 0.5
MAX_ENTROPY = 0.5

# Overlap rule (a tua regra)
OVERLAP_IOU = 0.05         # considera "sobrepostas" se IoU >= isto
# Nota: 0.15–0.30 costuma funcionar bem

# "Mais branco" dentro da bbox:
# (assume fundo escuro e dígitos claros)
USE_MEAN_WHITE = False       # True -> usa média; False -> usa soma total (mean*area)

#Samples
MAX_SAMPLES = 20  # 0 = corre todas


# ============================================================
# Modelo (igual à Tarefa 1)
# ============================================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# Utilitários
# ============================================================
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
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return float(inter / union)


def windows_for_image(W: int, H: int, win_sizes: List[int], stride: int):
    for s in win_sizes:
        if s > W or s > H:
            continue
        for y in range(0, H - s + 1, stride):
            for x in range(0, W - s + 1, stride):
                yield (x, y, s)


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def whiteness_score(img_np: np.ndarray, bbox) -> float:
    """
    Score de "branco" dentro da bbox.
    bbox: [x,y,w,h]
    Retorna:
      - média (0..255) se USE_MEAN_WHITE
      - ou soma total (mean*area) se não
    """
    x, y, w, h = bbox
    # bounds
    x2 = min(x + w, img_np.shape[1])
    y2 = min(y + h, img_np.shape[0])
    x = max(0, x)
    y = max(0, y)
    w = max(1, x2 - x)
    h = max(1, y2 - y)

    patch = img_np[y:y+h, x:x+w].astype(np.float32)
    mean_val = float(patch.mean())
    if USE_MEAN_WHITE:
        return mean_val
    return mean_val * float(w * h)


def overlap_keep_whitest(dets: List[Dict], img_np: np.ndarray, iou_thr: float) -> List[Dict]:
    """
    Se duas caixas têm IoU >= iou_thr:
      - manter a que tem maior whiteness_score
    Implementação tipo NMS, mas o critério de "melhor" é o branco, não o score do modelo.
    """
    if not dets:
        return dets

    # calcula whiteness e guarda
    for d in dets:
        d["white"] = whiteness_score(img_np, d["bbox"])

    # ordenar por "white" desc (mais branco primeiro)
    dets = sorted(dets, key=lambda x: x["white"], reverse=True)

    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou_xywh(best["bbox"], d["bbox"]) < iou_thr]

    # opcional: remover o campo "white" do output final (fica limpo)
    for d in kept:
        d.pop("white", None)

    return kept


def draw_detections(img: Image.Image, dets: List[Dict]) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)

    for d in dets:
        x, y, w, h = d["bbox"]
        lbl = int(d["label"])
        sc = float(d["score"])

        # cores fixas
        COLOR_NUMBER = (0, 255, 0)   # verde
        COLOR_SCORE = (255, 0, 0)    # vermelho
        COLOR_BOX = (255, 255, 255)  # branco

        # box
        draw.rectangle([x, y, x + w, y + h], outline=COLOR_BOX, width=2)

        # texto por cima
        ty = max(0, y - 14)
        draw.text((x, ty), str(lbl), fill=COLOR_NUMBER)
        draw.text((x + 12, ty), f"{sc:.2f}", fill=COLOR_SCORE)

    return out



# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_imgs = os.path.join(OUTPUT_DIR, "images")
    out_json = os.path.join(OUTPUT_DIR, "detections")
    os.makedirs(out_imgs, exist_ok=True)
    os.makedirs(out_json, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    model = CNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model: {MODEL_PATH}")

    tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    pngs = sorted(glob.glob(os.path.join(IMAGES_SOURCE, "*.png")))
    if len(pngs) == 0:
        raise FileNotFoundError(f"Não encontrei .png em: {IMAGES_SOURCE}")

    if MAX_SAMPLES and MAX_SAMPLES > 0:
        pngs = pngs[:MAX_SAMPLES]

    print(f"[INFO] Source: {IMAGES_SOURCE}")
    print(f"[INFO] Nº de imagens: {len(pngs)}")
    print(f"[INFO] STRIDE={STRIDE} WIN_SIZES={WIN_SIZES} BATCH={BATCH_SIZE}")
    print(f"[INFO] CONF={CONF_THRESH} MARGIN={MARGIN_THRESH} ENTROPY<={MAX_ENTROPY}")
    print(f"[INFO] OVERLAP_IOU={OVERLAP_IOU} keep=whitest (USE_MEAN_WHITE={USE_MEAN_WHITE})")

    for idx, png_path in enumerate(pngs, start=1):
        base = os.path.splitext(os.path.basename(png_path))[0]
        img = Image.open(png_path).convert("L")
        W, H = img.size
        img_np = np.array(img)  # uint8 0..255

        win_list = list(windows_for_image(W, H, WIN_SIZES, STRIDE))

        dets = []
        batch_tensors = []
        batch_meta = []

        def flush():
            nonlocal dets, batch_tensors, batch_meta
            if not batch_tensors:
                return

            x_tensor = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            for meta, pr in zip(batch_meta, probs):
                order = np.argsort(pr)[::-1]
                l1, l2 = int(order[0]), int(order[1])
                p1, p2 = float(pr[l1]), float(pr[l2])
                marg = p1 - p2
                ent = entropy(pr)

                if p1 >= CONF_THRESH and marg >= MARGIN_THRESH and ent <= MAX_ENTROPY:
                    x, y, s = meta
                    dets.append({
                        "label": l1,
                        "score": p1,
                        "bbox": [int(x), int(y), int(s), int(s)]
                    })

            batch_tensors = []
            batch_meta = []

        for (x, y, s) in win_list:
            crop = img.crop((x, y, x + s, y + s))
            tensor = tf(crop)
            batch_tensors.append(tensor)
            batch_meta.append((x, y, s))
            if len(batch_tensors) >= BATCH_SIZE:
                flush()
        flush()

        # ---- A tua regra: se há sobreposição, manter a que tem mais branco
        dets_kept = overlap_keep_whitest(dets, img_np, OVERLAP_IOU)

        # guardar json
        det_json_path = os.path.join(out_json, base + "_detections.json")
        with open(det_json_path, "w", encoding="utf-8") as f:
            json.dump(dets_kept, f, indent=2, ensure_ascii=False)

        # guardar imagem
        out_img = draw_detections(img, dets_kept)
        out_img_path = os.path.join(out_imgs, base + "_detections.png")
        out_img.save(out_img_path)

        print(f"[{idx}/{len(pngs)}] {base}: windows={len(win_list)} raw_dets={len(dets)} kept={len(dets_kept)}")

    print(f"[DONE] Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
