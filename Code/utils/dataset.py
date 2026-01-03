import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
import torch
import numpy as np
class Annotation:
    def __init__(self, filename):
        self.filename = filename
        tree = ET.parse(self.filename)
        root = tree.getroot()
        size = root.find('size')
        self.width = int(size.find('width').text)
        self.height = int(size.find('height').text)
        self.depth = int(size.find('depth').text)
        obj = root.find('object')
        bndbox = obj.find('bndbox')
        self.xmin = int(bndbox.find('xmin').text)
        self.ymin = int(bndbox.find('ymin').text)
        self.xmax = int(bndbox.find('xmax').text)
        self.ymax = int(bndbox.find('ymax').text)

class CatDogDataset(Dataset):
    def __init__(self, image_path, frame_path, mask_path):

        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = None
        self.frame_path = frame_path
        #self.label_path = 'logs/labels.txt'

        self.breeds_mapping = {}
        self.breed_num = {}
        self.num_breed = {}

        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(".jpg")]
        rows = []
        count = 0
        for image_file in image_files:

            image_name = os.path.splitext(image_file)[0]
            parts = image_name.split('_')
            breed_name = ' '.join(parts[:-1])
            if image_file[0].isupper():
                family_name = 'cat'
            else:
                family_name = 'dog'
            if breed_name not in self.breeds_mapping:
                self.breeds_mapping[breed_name] = family_name
                self.breed_num[breed_name] = count
                self.num_breed[count] = breed_name
                count += 1

            rows.append([image_name, breed_name, family_name])

        self.df_label = pd.DataFrame(rows, columns=["imagename", "breed", "family"])

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.df_label.shape[0]

    def __getitem__(self, idx):
        image_name, breed, family = self.df_label.iloc[idx]   

        image_path = os.path.join(self.image_path, image_name + '.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_path, image_name + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
        if mask is not None:
            mask = np.where(mask == 2, 0, mask)
            mask = np.where(mask == 3, 1, mask)
        #mask = (mask == 1).astype(np.float32)
        xml_path = os.path.join(self.frame_path, image_name + ".xml")
        if os.path.exists(xml_path):
            anno = Annotation(xml_path)
            bbox = [anno.xmin, anno.ymin, anno.xmax, anno.ymax]
        else:
            bbox = None

        if self.transform:
            transformed = self.transform(
                image=image,
                mask=mask,
                bboxes=[bbox] if bbox is not None else [],
                class_labels=["object"] if bbox is not None else []
            )
            image = transformed["image"]
            mask = transformed["mask"]
            bbox = transformed["bboxes"][0] if len(transformed["bboxes"]) > 0 else None
        label = [breed, family]
        return image, label, bbox, mask
    
    def get_breed_num(self, list_breed):
        outs = []
        for i in list_breed:
            outs.append(self.breed_num[i])
        return outs

def custom_classifier(batch):
    images, labels, families= [], [], []
    
    for item in batch:
        img, (lbl, fam), _, _ = item 
        images.append(img)
        families.append(0 if fam.lower() == 'cat' else 1) 
        labels.append(lbl)
    
    return (torch.stack(images), 
            (labels, torch.tensor(families)))

def custom_segmentation(batch):
    images, masks,labels, families= [], [], [], []
    
    for item in batch:
        img, (lbl, fam), _, mask = item 
        mask = mask.float()
        images.append(img)
        masks.append(mask)
        families.append(fam) 
        labels.append(lbl)
    
    return (torch.stack(images), (labels,families ), torch.stack(masks))