import os
import random
import torch
from torchvision import datasets
from torchvision.transforms import ToPILImage
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = 100
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or
                y1 + h1 < y2 or y2 + h2 < y1)

def generate_scene(mnist, num_digits, scale_range, save_path, idx):
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    boxes = []

    for _ in range(num_digits):
        digit, label = mnist[random.randint(0, len(mnist) - 1)]
        size = random.randint(*scale_range)
        digit = digit.resize((size, size))

        for _ in range(50):  # tentativas para evitar overlap
            x = random.randint(0, IMAGE_SIZE - size)
            y = random.randint(0, IMAGE_SIZE - size)
            new_box = (x, y, size, size)

            if not any(boxes_overlap(new_box, b[:4]) for b in boxes):
                canvas.paste(digit, (x, y))
                boxes.append((x, y, size, size, label))
                break

    img_path = os.path.join(save_path, f"{idx:06d}.png")
    ann_path = os.path.join(save_path, f"{idx:06d}.txt")

    canvas.save(img_path)
    with open(ann_path, "w") as f:
        for x, y, w, h, label in boxes:
            f.write(f"{label} {x} {y} {w} {h}\n")

def generate_dataset(version, num_images=5000):
    os.makedirs(f"data/scenes_{version}", exist_ok=True)

    mnist = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToPILImage(),
        source_url=MNIST_URL
    )

    if version == "A":
        num_digits = lambda: 1
        scale = (28, 28)
    elif version == "B":
        num_digits = lambda: 1
        scale = (22, 36)
    elif version == "C":
        num_digits = lambda: random.randint(3, 5)
        scale = (28, 28)
    elif version == "D":
        num_digits = lambda: random.randint(3, 5)
        scale = (22, 36)

    for i in tqdm(range(num_images), desc=f"Dataset {version}"):
        generate_scene(mnist, num_digits(), scale, f"data/scenes_{version}", i)

if __name__ == "__main__":
    for v in ["A", "B", "C", "D"]:
        generate_dataset(v, num_images=3000)
