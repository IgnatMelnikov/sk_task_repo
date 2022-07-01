from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

from dataset_creator import ImagesDataset
import os


# add to get computing device
def get_computing_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


default_device = get_computing_device()


# -----

def train_maps(n_epoch=100, device=default_device, load_state=False):
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=100)

    if load_state:
        ddpm.load_state_dict(torch.load("ddpm_maps.pth"))
    ddpm.to(device)

    # Added transforamtions and custom dataset creation
    tr = transforms.Compose(
        [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)
        ]
    )

    dataset = ImagesDataset(
        os.path.join(os.getcwd(), '..', 'EuroSAT/2750'),
        tr)
    # -----
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 64, 64), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_maps{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_maps.pth")


if __name__ == "__main__":
    train_maps()
