import os
import cv2
import xml.etree.ElementTree as ET
import pprint
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image
import matplotlib.pyplot as plt

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


def main():
    i = 4
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    data = get_data()
    sample = data[i]
    pprint.pprint(sample)
    # show_data(data[i])
    image = transform_image(sample)
    boxes = torch.tensor(sample["boxes"], dtype=torch.float)
    labels = torch.tensor(sample["labels"], dtype=torch.int64)
    d = {"boxes": boxes, "labels": labels}
    print(model([image], [d]))


if __name__ == "__main__":
    main()
