import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from modules import *

# Hyperparameters
nz = 100  # Size of latent vector (input to the generator)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
nc = 1    # Number of channels in the training images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_mu=True
# Create the generator and discriminator
netG = Generator(nz, ngf, nc)
netD = Discriminator(nc+1, ndf)

# Loss function and optimizers
criterion = nn.BCELoss()
criterion_mz = nn.MSELoss()
optimizer_G = optim.RMSprop(netG.parameters(), lr=0.00005)
optimizer_D = optim.RMSprop(netD.parameters(), lr=0.000005)

imgsize = 64
batch_size = 4
ising_data=np.load('../IsingData/ising_dataset2.npy')
temperature_label=np.load('../IsingData/labels_dataset2.npy')
dataset = TensorDataset(torch.tensor(ising_data,dtype=torch.float),torch.tensor(temperature_label,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

num_epochs = 300
real_label = 1.0
fake_label = 0.0
netG.to(device)
netD.to(device)
wf=open('./res/log.csv','w')
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        optimizer_D.zero_grad()
        
        real_data = data[0].to(device)
        temper=data[1].to(device)
        batch_size = real_data.size(0)
        mz_real = torch.mean(real_data, (-1,-2,-3))
        label = torch.full((batch_size,), real_label, device=device)
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_data = netG(noise,temper)
        loss_D = -torch.mean(netD(real_data.unsqueeze(1),temper)) + torch.mean(netD(fake_data,temper))
        
        loss_D.backward()
        optimizer_D.step()
        
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
            
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_data = netG(noise,temper)
        
        if loss_mu:
            mz_fake = torch.mean(fake_data, (-1,-2,-3))
            loss_G = -torch.mean(netD(fake_data,temper)) + criterion_mz(mz_real.float().to(device), mz_fake.float().to(device))
        else:
            loss_G = -torch.mean(netD(fake_data,temper))
            
        loss_G.backward()
        optimizer_G.step()

    wf.write(f"{epoch}: Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
    torch.save(netG.state_dict(), './res/g_'+str(epoch)+'.pth')
    torch.save(netD.state_dict(), './res/d_'+str(epoch)+'.pth')

