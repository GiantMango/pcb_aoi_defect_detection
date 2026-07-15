import cv2
import xml.etree.ElementTree as ET
import pprint


def get_data():
    data = []
    with open("train_data/index.txt", "r") as f:
        for line in f:
            dict = {}
            filename, annotation_filename = line.strip().split(" ")
            dict["image"] = filename
            dict["annotation"] = annotation_filename
            data.append(dict)

    for d in data:
        tree = ET.parse(f"train_data/{d['annotation']}")
        root = tree.getroot()
        d["defects"] = []

        for object in root.findall("object"):
            dict = {}
            dict["label"] = object.find("name").text
            dict["xmin"] = int(object.find("bndbox")[0].text)
            dict["ymin"] = int(object.find("bndbox")[1].text)
            dict["xmax"] = int(object.find("bndbox")[2].text)
            dict["ymax"] = int(object.find("bndbox")[3].text)
            d["defects"].append(dict)

    return data


def main():
    data = get_data()
    pprint.pprint(data[2])
    current = data[2]
    image = cv2.imread(f"train_data/{current['image']}")
    for d in current["defects"]:
        cv2.rectangle(
            image,
            (d["xmin"], d["ymin"]),
            (d["xmax"], d["ymax"]),
            (255, 0, 0),
            2,
        )
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
