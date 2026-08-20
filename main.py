import os
import cv2
import xml.etree.ElementTree as ET
import pprint
import torch
import torch.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import random
import time

DATA_DIR = "train_data"
LABEL_TO_ID = {"background": 0, "Bad_podu": 1, "Bad_qiaojiao": 2}
NUM_LABELS = len(LABEL_TO_ID)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        with open(os.path.join(DATA_DIR, "index.txt"), "r") as f:
            for line in f:
                boxes = []
                labels = []

                filename, annotation_filename = line.strip().split(" ")

                tree = ET.parse(os.path.join(DATA_DIR, annotation_filename))
                root = tree.getroot()

                for object in root.findall("object"):
                    labels.append(LABEL_TO_ID[object.find("name").text])
                    boxes.append(
                        [
                            int(object.find("bndbox").find("xmin").text),
                            int(object.find("bndbox").find("ymin").text),
                            int(object.find("bndbox").find("xmax").text),
                            int(object.find("bndbox").find("ymax").text),
                        ]
                    )
                self.data.append(
                    {
                        "image_path": os.path.join(DATA_DIR, filename),
                        "boxes": boxes,
                        "labels": labels,
                    }
                )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load images
        image = decode_image(self.data[idx]["image_path"])
        labels = self.data[idx]["labels"]
        boxes = self.data[idx]["boxes"]
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(image), dtype=torch.float
        )
        target["labels"] = torch.tensor(labels, dtype=torch.int64)

        if self.transform is not None:
            image = self.transform(image)

        return (image, target)


def collate_wrapper(batch):
    transposed_data = [list(row) for row in zip(*batch)]
    return (transposed_data[0], transposed_data[1])


def get_transform(train):
    if train:
        transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
    return transforms


def prepare_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=NUM_LABELS
    )
    return model


def main():
    num_epochs = 2

    training_data = CustomImageDataset(get_transform(True))
    train_dataloader = DataLoader(
        training_data, batch_size=2, shuffle=True, collate_fn=collate_wrapper
    )

    model = prepare_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.to(DEVICE)

    # Training
    start = time.time()
    for e in range(num_epochs):
        epoch_loss = 0
        for b in range(len(train_dataloader)):

            images, targets = next(iter(train_dataloader))

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss = model(images, targets)
            total_loss = sum(l for l in loss.values())
            total_loss.backward()
            optimizer.step()

        epoch_loss += total_loss.item()
        print(f"Epoch {e+1}/{num_epochs}: epoch_loss {epoch_loss:>7f}")
    end = time.time()
    print(f"Average Loss: {epoch_loss/num_epochs:.6f}")
    print(f"Average Time: {(end-start)/num_epochs} s")


if __name__ == "__main__":
    main()
