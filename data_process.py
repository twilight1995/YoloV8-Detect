import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm


def get_xml_dict(xml_root):
    class_info = {}
    for root, dirs, files in os.walk(xml_root):
        if files is not None:
            for file in files:
                if file.endswith('.xml'):
                    xml_path = os.path.join(root, file)
                    # Load the XML annotation data
                    tree = ET.parse(xml_path)
                    treeroot = tree.getroot()

                    for obj in treeroot.findall('object'):
                        name = obj.find('name').text
                        if name not in class_info:
                            class_info[name] = 1
                        else:
                            class_info[name] += 1
    return class_info


def parse_xml(xml_path):
    # Load the XML annotation data
    tree = ET.parse(xml_path)
    treeroot = tree.getroot()
    return treeroot.findall('object')


def convert(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    width = xmax - xmin
    height = ymax - ymin
    return center_x, center_y, width, height


def get_all_files(img_xml_root, file_type='.xml'):
    img_paths = []
    xml_paths = []
    # get all files
    for root, dirs, files in os.walk(img_xml_root):
        if files is not None:
            for file in files:
                if file.endswith(file_type):
                    file_path = os.path.join(root, file)
                    if file_type in ['.xml']:
                        img_path = file_path[:-4] + '.jpg'
                        if os.path.exists(img_path):
                            xml_paths.append(file_path)
                            img_paths.append(img_path)
                    elif file_type in ['.jpg']:
                        xml_path = file_path[:-4] + '.xml'
                        if os.path.exists(xml_path):
                            img_paths.append(file_path)
                            xml_paths.append(xml_path)

                    elif file_type in ['.json']:
                        img_path = file_path[:-5] + '.jpg'
                        if os.path.exists(img_path):
                            img_paths.append(img_path)
                            xml_paths.append(file_path)
    return img_paths, xml_paths


def train_test_split(img_paths, xml_paths, test_size=0.2):
    img_xml_union = list(zip(img_paths, xml_paths))
    np.random.shuffle(img_xml_union)
    train_set = img_xml_union[:int(len(img_xml_union) * (1 - test_size))]
    test_set = img_xml_union[int(len(img_xml_union) * (1 - test_size)):]
    return train_set, test_set


def convert_annotation(img_xml_set, classes, save_path, is_train=True):
    os.makedirs(os.path.join(save_path, 'images', 'train' if is_train else 'val'), exist_ok=True)
    img_root = os.path.join(save_path, 'images', 'train' if is_train else 'val')
    os.makedirs(os.path.join(save_path, 'labels', 'train' if is_train else 'val'), exist_ok=True)
    txt_root = os.path.join(save_path, 'labels', 'train' if is_train else 'val')
    for item in tqdm(img_xml_set):
        img_path = item[0]
        txt_file_name = os.path.split(img_path)[-1][:-4] + '.txt'
        shutil.copy(img_path, img_root)
        img = cv2.imread(img_path)
        size = (img.shape[1], img.shape[0])
        xml_path = item[1]
        dict_info = parse_xml(xml_path)

        yolo_infos = []
        for obj in dict_info:
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            center_x, center_y, w, h = convert(xmin, ymin, xmax, ymax)
            cat_box = [str(classes.index(name)), str(float(center_x) / 1000.0), str(float(center_y) / 1000.0),
                       str(float(w) / 1000.0), str(float(h) / 1000.0)]
            yolo_infos.append(' '.join(cat_box))

        if len(yolo_infos) > 0:
            with open(os.path.join(txt_root, txt_file_name), 'w', encoding='utf_8') as f:
                for info in yolo_infos:
                    f.writelines(info)
                    f.write('\n')


if __name__ == '__main__':
    xml_root = r"D:\project\data\yolov8\img_xml"
    save_path = "./dataset"
    os.makedirs(save_path, exist_ok=True)
    class_info = get_xml_dict(xml_root)
    classes = list(class_info.keys())
    with open("./classes.txt", 'w', encoding='utf_8') as f:
        for idx, class_name in enumerate(classes):
            f.writelines(f"{idx}: {class_name}")
            f.write('\n')
    res = get_all_files(xml_root, file_type='.xml')
    train_set, test_set = train_test_split(res[0], res[1], test_size=0.2)
    convert_annotation(train_set, classes, save_path, is_train=True)
    convert_annotation(test_set, classes, save_path, is_train=False)
