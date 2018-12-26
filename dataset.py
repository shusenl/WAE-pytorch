"""dataset.py"""

from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

class CustomImageList(Dataset):
    def __init__(self, fileListName, transforms = None):
        self.data = []
        self.transforms = transforms
        ## load image path / label pair
        with open(fileListName) as f:
            self.data = [x.split() for x in f.readlines()]

    def __getitem__(self, index):
        path, label = self.data[index]
        img = Image.open(path)
        if self.transforms is not None:
            img = self.transforms(img)
        #print ("label: ",label)
        return img, torch.tensor(int(label))

    def __len__(self):
        return len(self.data)

    def test(self):
        for item in self.data:
           print("label:", item[1])

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers

    if name.lower() == 'celeba':
        root = Path(dset_dir).joinpath('CelebA_trainval')
        transform = transforms.Compose([
            transforms.CenterCrop((240, 240)),
            # transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader

if __name__ == '__main__':
    pass
