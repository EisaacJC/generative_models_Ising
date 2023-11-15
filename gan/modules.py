import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2*nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.conv1=nn.Linear(in_features=1,out_features=nz)
        self.relu1=nn.ReLU()
        self.conv2=nn.Linear(in_features=nz,out_features=nz)
        self.relu2=nn.ReLU()
        
                    

    def forward(self, z, temp):
        adder=self.relu1(self.conv1(torch.reshape(temp,(-1,1))))
        adder2=self.relu2(self.conv2(adder)).unsqueeze(-1).unsqueeze(-1)
        combined_input=torch.cat([z,adder2],1)
        res=self.main(combined_input)
        return res

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        #self.embedding = nn.Linear(1, 64*64)
        self.conv1=nn.Linear(in_features=1,out_features=64*64)
        self.relu1=nn.ReLU()
        self.conv2=nn.Linear(in_features=64*64,out_features=64*64)
        self.relu2=nn.ReLU()
        
    def forward(self, img, temp):
        #temp_embedding = self.embedding(temp).view(-1,1, 64, 64)
        adder=self.relu1(self.conv1(torch.reshape(temp,(-1,1))))
        temp_embedding=self.relu2(self.conv2(adder)).view(-1,1, 64, 64)
        combined_input = torch.cat([img, temp_embedding], 1)
        res=self.main(combined_input)
        return res