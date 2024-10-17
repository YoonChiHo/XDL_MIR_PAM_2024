# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:37:33 2022

@author: Sampa
"""
import itertools
import Models 
from Dataset_cycleGAN import ImageDataset
from pytorch_msssim import ssim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torch.cuda import FloatTensor
import torch

from PIL import Image
from pytorch_msssim import ssim
import torch.nn as nn
import glob
import os

from Code_Utils import ReplayBuffer
from Code_Utils import LambdaLR
from Code_Utils import Logger
from Code_Utils import weights_init_normal

def tensor2image(tensor, type):
    if type == 'HE': # for HE image
        image = 127.5*(tensor[0].cpu().detach().float().numpy() + 1.0)
        return image.astype(np.uint8)
    else: # for PA image
        image = ((tensor[0].cpu().detach().float().numpy() * (-0.5)) + 0.5)*255
        return image.astype(np.uint8)
def display_image_test(images, name):
    imgs = images.squeeze(dim=0)
    imgs_np = tensor2image(imgs, type='HE')
    imgs_im = Image.fromarray(imgs_np)
    imgs_im.save("%s.png" % (name))

epoch = 0
n_epochs = 300
batchSize = 4#32 #3090 1060x1->x 3090x1->bs8 ncpu32   cpu->bs4 ncpu x
lr = 0.0002
decay_epoch = 50
size = 256  # Original used 256
input_nc = 1
output_nc = 1
NUM_WORKER =0
threshold_A = 90
threshold_B = 60

# Early Stopping with Saliency Loss
early_saliency = 100000 # Saliency loss lim for early stopping. 
early_warmup = 0 # Warup epochs for early stopping

netG_A2B = Models.Generator(input_nc, output_nc)
netG_B2A = Models.Generator(output_nc, input_nc)
netD_A = Models.Discriminator(input_nc)
netD_B = Models.Discriminator(output_nc)
device = torch.device("cuda")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # "0, 1, 2, 3"
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netG_A2B = torch.nn.DataParallel(netG_A2B).to(device)
    netG_B2A = torch.nn.DataParallel(netG_B2A).to(device)
    netD_A = torch.nn.DataParallel(netD_A).to(device)
    netD_B = torch.nn.DataParallel(netD_B).to(device)
else:
    netG_A2B = netG_A2B.to(device)
    netG_B2A = netG_B2A.to(device)
    netD_A = netD_A.to(device)
    netD_B = netD_B.to(device)
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
L1_function = torch.nn.L1Loss()
optimizer_G = Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=lr, betas=(0.5, 0.999))
optimizer_D_A = Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
Tensor = FloatTensor 
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

data_dir_train = "./Data"
dataroot_train_LR = data_dir_train  + '/Train_LR_256'
dataroot_train_HR = data_dir_train  + '/Train_gHR_256'
n_list_train_LR = sorted(glob.glob('%s/*.png'%(dataroot_train_LR)))
n_list_train_HR = sorted(glob.glob('%s/*.png'%(dataroot_train_HR)))
print(len(n_list_train_LR))
print(len(n_list_train_HR))

transforms_PA = [ transforms.ToPILImage(),
                      transforms.Resize((256,256)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5), (0.5)) ]
    
transforms_HE = [ transforms.ToPILImage(),
                      transforms.Resize((256,256)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),    
                      transforms.Normalize((0.5), (0.5)) ]          
train_dataloader = DataLoader(ImageDataset(dataroot_train_LR, dataroot_train_HR, transforms_PA=transforms_PA, transforms_HE=transforms_HE, train=True, unaligned=True), 
                            batch_size=batchSize, shuffle=True, num_workers=NUM_WORKER)
print(len(train_dataloader)  )   
 
logger = Logger(n_epochs, len(train_dataloader))
for epoch in range(0, n_epochs):
    saliency_loss = []
    for i, batch in enumerate(train_dataloader):
            # Set model input
        real_A = Variable(batch["PA"]).cuda() # [1, 3, 256, 256]
        real_B = Variable(batch["HE"]).cuda() # [1, 3, 256, 256]
    
            # mask generation
        real_A_class = batch["PA_class"]
        real_B_class = batch["HE_class"]
        optimizer_G.zero_grad()

        fake_B = netG_A2B(real_A) 
        pred_fake_B = netD_B(fake_B)
        recovered_A = netG_B2A(fake_B)

        fake_A = netG_B2A(real_B)
        pred_fake_A = netD_A(fake_A)
        recovered_B = netG_A2B(fake_A)


        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)

        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        real_A2 = torch.cat([real_A.cuda(),real_A.cuda(),real_A.cuda()], dim=1)
        fake_A2 = torch.cat([fake_A.cuda(),fake_A.cuda(),fake_A.cuda()], dim=1)
        fake2_A = netG_A2B(fake_B)
        fake2_B = netG_B2A(fake_A)
        loss_ssimA =1-ssim(real_A.cpu()+1, fake2_A.cpu()+1, data_range=2, size_average=True)
        loss_ssimB =1-ssim(real_B.cpu()+1, fake2_B.cpu()+1, data_range=2, size_average=True)
        loss_ssim = loss_ssimA + loss_ssimB
        real_A_mean = torch.mean(real_A,dim=1,keepdim=True)
        real_B_mean = torch.mean(real_B,dim=1,keepdim=True)
        fake_A_mean = torch.mean(fake_A,dim=1,keepdim=True)
        fake_B_mean = torch.mean(fake_B,dim=1,keepdim=True)
        real_A_normal = (real_A_mean - (threshold_A/127.5-1))*100
        real_B_normal = (real_B_mean - (threshold_B/127.5-1))*100

        fake_A_normal = (fake_A_mean - (threshold_B/127.5-1))*100
        fake_B_normal = (fake_B_mean - (threshold_A/127.5-1))*100

        real_A_sigmoid = torch.sigmoid(real_A_normal)
        real_B_sigmoid = torch.sigmoid(real_B_normal)

        fake_A_sigmoid = torch.sigmoid(fake_A_normal)
        fake_B_sigmoid = torch.sigmoid(fake_B_normal)
        content_loss_A = L1_function( real_A_sigmoid , fake_B_sigmoid )
        content_loss_B = L1_function( fake_A_sigmoid , real_B_sigmoid )

        content_loss = (content_loss_A + content_loss_B)
       
        saliency_loss.append(content_loss.detach().cpu())

        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_ssim + content_loss
        loss_G.backward()
        optimizer_G.step()
            #############################
    
            ###### Discriminator A ######
        optimizer_D_A.zero_grad()
            
            # Real loss
        pred_real = netD_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real, target_real)
    
            # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake_A = criterion_GAN(pred_fake, target_fake)
    
            # Total loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A)*0.5
        loss_D_A.backward()
    
        optimizer_D_A.step()
            #############################
            
            ###### Discriminator B ######
        optimizer_D_B.zero_grad()
    
            # Real loss
        pred_real = netD_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real, target_real)
            
            # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake_B = criterion_GAN(pred_fake, target_fake)
    
            # Total loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B)*0.5
        loss_D_B.backward()
    
        optimizer_D_B.step()
            #############################
    
            # Progress report
        Loss = logger.log(losses = {'loss_G': loss_G, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A), 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B), 'loss_D_real': (loss_D_real_A + loss_D_real_B), 'loss_D_fake': (loss_D_fake_A + loss_D_fake_B)})
            # Display images

    # Early Stopping with Saliency Loss
    def vis_diff(img_A, img_B, name):

        image1 = img_A.numpy() 
        image2 = img_B.numpy()  
        difference = np.abs(image1 - image2)
        threshold = 0.1  
        highlight_mask = difference > threshold
        image1_rgb = np.stack([image1]*3, axis=-1) 
        image1_rgb[highlight_mask] = [1, 0, 0] 
        image1_rgb_uint8 = (image1_rgb * 255).astype(np.uint8)
        output_image = Image.fromarray(image1_rgb_uint8)

        output_image.save(name)
    if np.mean(saliency_loss) > early_saliency and epoch >= early_warmup:
        print(f'ERROR: Epoch {epoch} saliency loss {np.mean(saliency_loss):.4f} is over than limitation {early_saliency}')
        vis_diff(real_A_sigmoid[0,0,...].detach().cpu(), fake_B_sigmoid[0,0,...].detach().cpu(), f'./checkpoint/Earlystop_e{epoch}_{np.mean(saliency_loss):.4f}.png')
        break
    
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
        
    
    torch.save(netG_A2B.state_dict(), "./checkpoint/checkpointG_A2B_HR_XCG.pt")
    
