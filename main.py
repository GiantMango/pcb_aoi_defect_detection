import os
import cv2
import xml.etree.ElementTree as ET
import pprint

DATA_DIR = "train_data"
LABEL_TO_ID = {"Bad_podu": 0, "Bad_qiaojiao": 1}


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


def main():
    i = 4
    data = get_data()
    pprint.pprint(data[i])
    show_data(data[i])


if __name__ == "__main__":
    main()
