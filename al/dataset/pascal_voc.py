import os
import xml.etree.ElementTree as ET
from PIL import Image

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Dataset

from .active_dataset import ActiveDataset, MaskDataset
from ..helpers.constants import DATA_ROOT
from ..model.model_zoo.ssd import Container, get_transforms, get_transforms_semantic



class PascalVOCObjectDataset(ActiveDataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, indices, n_init=100, output_dir=None, train=True, year='2012', cfg=None):
        self.data_dir = os.path.join(DATA_ROOT, f'VOCdevkit/VOC{year}')
        self.cfg = cfg
        self.init_dataset = self._get_initial_dataset(train, year)
        super().__init__(self.get_dataset(indices), n_init=n_init, output_dir=output_dir)


    def _get_initial_dataset(self, train=True, year='2012'):
        transform, target_transform = get_transforms(self.cfg, train)
        if train:
            image_set = 'train'
        else:
            image_set = 'val'
        if not os.path.exists(self.data_dir):
            torchvision.datasets.VOCDetection(root=DATA_ROOT, year=year, image_set=image_set, download=True)
        split = 'train' if train else 'val'
        if train: return VOCDataset(self.data_dir, split, transform=transform, target_transform=target_transform, keep_difficult=not train)
        else: return VOCDataset(self.data_dir, split, transform=transform, target_transform=target_transform, keep_difficult=not train)

    def get_dataset(self, indices):
        return MaskDataset(self.init_dataset, indices)





class PascalVOCSemanticDataset(ActiveDataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, indices, n_init=100, output_dir=None, train=True, year='2012', cfg=None):
        self.data_dir = os.path.join(DATA_ROOT, f'VOCdevkit/VOC{year}')
        self.cfg = cfg
        self.init_dataset = self._get_initial_dataset(train, year)
        super().__init__(self.get_dataset(indices), n_init=n_init, output_dir=output_dir)

    def _get_initial_dataset(self, train=True, year='2012'):
        transform, target_transform = get_transforms_semantic(self.cfg)
        if train:
            image_set = 'train'
        else:
            image_set = 'val'
        if not os.path.exists(self.data_dir):
            torchvision.datasets.VOCSegmentation(root=DATA_ROOT, year=year, image_set=image_set, download=True)
        split = 'train' if train else 'val'
        return VOCDatasetSemantic(self.data_dir, split, transform=transform, target_transform=target_transform, keep_difficult=not train)

    def get_dataset(self, indices):
        return MaskDataset(self.init_dataset, indices)

    def set_validation_dataset(self, dataset):
        self.val_dataset = dataset

    def get_validation_dataset(self):
        return self.val_dataset



class VOCDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for _, line in enumerate(f):
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image


class SmallVOCDataset(VOCDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 30

class VOCDatasetSemantic(VOCDataset):

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Segmentation", "%s.txt" % self.split)
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        label_image = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image = self.transform(image)
        print(image.shape)
        if self.target_transform:
            label_image = self.target_transform(label_image)
        return image, label_image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for _, line in enumerate(f):
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "SegmentationClass", "%s.png" % image_id)
        image = Image.open(annotation_file)
        return image

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        return image

        