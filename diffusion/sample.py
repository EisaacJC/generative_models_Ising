import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from modules import UNet_conditional, Diffusion
from torch.utils.data import DataLoader, TensorDataset
import sys

#Set up training parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgsize = 64
n_sample=1000
batch_size = 4
nloop=int(n_sample/batch_size)
lr=3e-4
tpred=float(sys.argv[1])

model = UNet_conditional().to(device)
model.load_state_dict(torch.load('trained_model.pt'))
diffusion = Diffusion(img_size=imgsize, device=device)
samples=np.empty((0,1,64,64))
for i1 in range(nloop):
    sampled_images = diffusion.sample(model, n=4,labels=torch.tensor([tpred,tpred,tpred,tpred],dtype=torch.float,device=device))
    simg_np=sampled_images.to('cpu').numpy()
    samples=np.vstack((samples,simg_np))
np.save("samples_"+str(tpred)+".npy",samples)