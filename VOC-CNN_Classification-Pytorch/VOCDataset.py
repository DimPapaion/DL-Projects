import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class VOCData(Dataset):
    def __init__(self, data_dir, image_dir, data_type, transform=None):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.data_type = data_type
        self.transform = transform
        self.images = os.listdir(image_dir)

        self.__init_classes()
        self.names, self.labels = self.__dataset_info()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.names[index ] +'.jpg')
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            label = self.labels[index]
        return image, label

    def __dataset_info(self):
        # annotation_files = os.listdir(self.data_path+'/Annotations')
        with open(self.data_dir +'/ImageSets/Main/ ' +self.data_type +'.txt') as f:
            annotations = f.readlines()
        annotations = [n[:-1] for n in annotations]
        names = []
        labels = []

        for af in annotations:
            filename = os.path.join(self.data_dir ,'Annotations' ,af)
            tree = ET.parse(filename +'.xml')
            objs = tree.findall('object')
            num_objs = len(objs)
            boxes_cl = np.zeros((num_objs), dtype=np.int32)

            for ix, obj in enumerate(objs):
                cls = self.class_to_ind[obj.find('name').text.lower().strip()]
                boxes_cl[ix] = cls
            lbl = np.zeros(self.num_classes)
            lbl[boxes_cl] = 1
            labels.append(lbl)
            names.append(af)
        return np.array(names), np.array(labels).astype(np.float32)

    def __init_classes(self):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes  = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

