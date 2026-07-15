import cv2
import xml.etree.ElementTree as ET
import pprint


def get_data():
    paths = []
    data = []
    with open("train_data/index.txt", "r") as f:
        for line in f:
            dict = {}
            filename, annotation_filename = line.strip().split(" ")
            dict["image_path"] = filename
            dict["annotation_path"] = annotation_filename
            paths.append(dict)

    for path in paths:
        tree = ET.parse(f"train_data/{path['annotation_path']}")
        root = tree.getroot()
        dict = {"labels": [], "boxes": []}
        for object in root.findall("object"):
            dict["labels"].append(object.find("name").text)
            dict["boxes"].append(
                [
                    int(object.find("bndbox").find("xmin").text),
                    int(object.find("bndbox").find("ymin").text),
                    int(object.find("bndbox").find("xmax").text),
                    int(object.find("bndbox").find("ymax").text),
                ]
            )

        data.append(
            {
                "image_path": path["image_path"],
            }
            | dict
        )

    return data


def main():
    i = 4
    data = get_data()
    pprint.pprint(data[i])
    current = data[i]
    image = cv2.imread(f"train_data/{current['image_path']}")
    for box in current["boxes"]:
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


if __name__ == "__main__":
    main()
