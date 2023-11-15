import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from modules import *

# Hyperparameters
nz = 100
ngf = 64
ndf = 64
nc = 1

# Create the generator and discriminator
netG = Generator(nz, ngf, nc)
netD = Discriminator(nc+1, ndf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_sample = 1000
real_label = 1.0
fake_label = 0.0
netG.to(device)
netD.to(device)
netG.load_state_dict(torch.load('g_trained.pth'))
temper=float(sys.argv[1])
samples=np.empty((0,1,64,64))
for iepoch in range(num_sample):
    t1=torch.tensor([temper]).view(1,-1).to(device)
    n1 = torch.randn(1, nz, 1, 1, device=device)
    fake1 = netG(n1,t1)
    fake1_np=fake1.to('cpu').detach().numpy()
    samples=np.vstack((samples,fake1_np))
np.save("samples_"+str(temper)+".npy",samples)

