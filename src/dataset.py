import os
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from image_processor import extract_patch


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, path_dataset, size=160):
        self.path = path_dataset
        self.imgs = os.listdir(path_dataset)

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image =  Image.open(os.path.join(self.path, self.imgs[idx]))
        inputs = self.transforms(image)

        return inputs, "dummy"

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, trg_img, window_size=160, stride=96):
        self.imgs = extract_patch(trg_img, window_size, stride)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        inputs = self.transforms(self.imgs[idx])

        return inputs, "dummy"

