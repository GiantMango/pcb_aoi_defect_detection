import os
import cv2
import xml.etree.ElementTree as ET
import pprint
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import random
import time

DATA_DIR = "train_data"
LABEL_TO_ID = {"background": 0, "Bad_podu": 1, "Bad_qiaojiao": 2}


def get_data():
    data = []
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
            data.append({"image_path": filename, "boxes": boxes, "labels": labels})

    return data


def show_data(sample):
    image = cv2.imread(os.path.join(DATA_DIR, sample["image_path"]))
    for box in sample["boxes"]:
        cv2.rectangle(
            image,
            (box[0], box[1]),
            (box[2], box[3]),
            (255, 0, 0),
            2,
        )
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform_image(image):
    pil_image = Image.open(os.path.join(DATA_DIR, image["image_path"]))

    transforms = v2.Compose(
        [
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        ]
    )

    torch_image = transforms(pil_image)

    return torch_image


def prepare_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)
    return model


def main():
    model = prepare_model()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    data_dict = get_data()
    train_size = round(0.85 * len(data_dict))
    train_dict = data_dict[:train_size]
    validate_dict = data_dict[train_size:]

    batch_size = 1
    batches = round(train_size / batch_size)
    epoch_size = 2

    for e in range(epoch_size):
        start = time.time()
        random.shuffle(train_dict)
        epoch_loss = 0

        for b in range(batches):
            images = []
            targets = []

            for i in range(batch_size):
                d = {}
                data = train_dict[b * batch_size + i]
                images.append(transform_image(data))
                d["boxes"] = torch.tensor(data["boxes"], dtype=torch.float)
                d["labels"] = torch.tensor(data["labels"], dtype=torch.int64)
                targets.append(d)

            optimizer.zero_grad()
            loss = model(images, targets)
            total_loss = sum(l for l in loss.values())
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            total_loss_ = total_loss.item()
            print(f"batch {b + 1}/{batches}: total loss {total_loss_:>7f}")

        end = time.time()
        average_loss = epoch_loss / batches
        print(
            f"Epoch {e + 1}/{epoch_size}: average loss {average_loss:.6f} | {end-start} s"
        )

        torch.save(
            {
                "epoch": e + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "average_loss": average_loss,
            },
            f"pcb_checkpoint_epoch_{e + 1}.pth",
        )


if __name__ == "__main__":
    main()
