# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:30:50 2022

@author: Sampa
"""
import Models 
from Dataset_cycleGAN import TestDataset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
import glob
import random
from Code_Utils import tensor2image

size = 256 # Original used 256
input_nc = 1
output_nc = 1
NUM_WORKER =0
def display_image_test(images, name):
    imgs = images.squeeze(dim=0)
    imgs_np = tensor2image(imgs, type='HE')
    imgs_im = Image.fromarray(imgs_np)
    imgs_im.save("test/DL_HR_Test_L/DL_%s.png" % (name))
def display_image_test_A(images, name):
    imgs = images.squeeze(dim=0)
    imgs_np = tensor2image(imgs, type='PA')
    imgs_im = Image.fromarray(imgs_np)
def tensor2image(tensor, type):
    if type == 'HE': # for HE image
        image = 127.5*(tensor[0].cpu().detach().float().numpy() + 1.0)
        return image.astype(np.uint8)
    else: # for PA image
        image = ((tensor[0].cpu().detach().float().numpy() * (-0.5)) + 0.5)*255
        return image.astype(np.uint8)
def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    np.random.seed(0)
    random.seed(0)
    Tensor = torch.Tensor
    data_dir_train = "./test"
    netG_A2B = Models.Generator(input_nc, output_nc)
    netG_B2A = Models.Generator(output_nc, input_nc)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG_A2B = torch.nn.DataParallel(netG_A2B).cuda()
        netG_B2A = torch.nn.DataParallel(netG_B2A).cuda()
        
    else:
        netG_A2B = netG_A2B.cuda()
        netG_B2A = netG_B2A.cuda()
    
    netG_A2B.load_state_dict(torch.load('./checkpoint/checkpointG_A2B_HR_XCG.pt'))
    transforms_test = [ transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)) 
                    ]
    dataset_LR = data_dir_train  + '/Test_L'    
    n_list_train_LR = sorted(glob.glob('%s/*.png'%(dataset_LR)))
    print("Number of LR Samples in Test ",len(n_list_train_LR))      
    Test_dataloader = DataLoader(TestDataset(dataset_LR,transforms_PA=transforms_test), 
                     batch_size=1, shuffle=False, num_workers=NUM_WORKER)  
    for i, batch in enumerate(Test_dataloader):
        # Set model input
        real_A = Variable(batch["LR"].type(Tensor)) 
        real_A_class = batch["LR_class"]
        name = batch["PA_name"]
        fake_B = netG_A2B(real_A) 

        for idx in range(fake_B.shape[0]):
            print(name[idx])
            display_image_test(fake_B[idx:idx+1], name[idx])  
if __name__ == '__main__':
    main()    