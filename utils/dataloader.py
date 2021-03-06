import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import glob
import json
from pycocotools.coco import COCO
import numpy as np

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, json_file, trainsize):
        self.trainsize = trainsize

        self.images = []
        self.gts = []
        coco = COCO(json_file)
        img_ids = coco.getImgIds()
        for i in img_ids:
            info = coco.loadImgs([i])[0]
            self.images.append(image_root + info['file_name'])
            self.gts.append(gt_root + info['file_name'])
        print(('all train images number %d') % len(self.images))
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        #print(1, np.array(gt).sum(), np.array(gt).shape)
        gt = np.array(gt) * 255
        gt = Image.fromarray(gt.astype('uint8')).convert('L')
        gt = self.gt_transform(gt)
        #print(2, gt.sum(), gt.shape)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            #print(img.size, gt.size)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, json_file, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset(image_root, gt_root, json_file, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, json_file, testsize):
        self.testsize = testsize
        annotations = json.load(open(json_file))
        self.images = []
        self.gts = []
        self.names = []
        self.images = []
        self.gts = []
        coco = COCO(json_file)
        img_ids = coco.getImgIds()
        for i in img_ids:
            info = coco.loadImgs([i])[0]
            self.images.append(image_root + info['file_name'])
            self.gts.append(gt_root + info['file_name'])
            self.names.append(info['file_name'])

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.names[self.index]
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
