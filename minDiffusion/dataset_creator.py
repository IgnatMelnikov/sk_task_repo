import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader


class ImagesDataset(Dataset):
    def __init__(self, data_root, trans=None):
        self.samples = []
        self.trans = trans
        for land_type in os.listdir(data_root):
            land_type_folder = os.path.join(data_root, land_type)
            for img_path in os.listdir(land_type_folder):
                img = Image.open(os.path.join(land_type_folder, img_path))
                if img is not None:
                    self.samples.append(
                        transforms.functional.to_tensor(img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.trans:
            sample = self.trans(sample)
        return sample
