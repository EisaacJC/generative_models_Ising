import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from modules import UNet_conditional, Diffusion
from torch.utils.data import DataLoader, TensorDataset


#Set up training parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgsize = 64
batch_size = 4
nepoch=300
nfreq=10
lr=3e-4

#Load Ising data
ising_data=np.load('../IsingData/ising_dataset2.npy')
ising_data.shape = -1, 1, imgsize, imgsize
temperature_label=np.load('../IsingData/labels_dataset2.npy')
dataset = TensorDataset(torch.tensor(ising_data,dtype=torch.float),torch.tensor(temperature_label,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

model = UNet_conditional().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
mse = nn.MSELoss()
diffusion = Diffusion(img_size=imgsize, device=device)

for epoch in range(nepoch):
    pbar = tqdm(dataloader)
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)
        predicted_noise = model(x_t, t, labels)
        loss = mse(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(MSE=loss.item())

    if epoch % nfreq == 0:
        torch.save(model.state_dict(), f"{epoch}ckpt.pt")

