# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:30:50 2022

@author: Sampa
"""
import Models 
from Dataset_cycleGAN_HE import TestDataset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
import glob
import random
import os

result_path = 'test/DL_HE_Test_H'
os.makedirs(f'./{result_path}', exist_ok=True)
os.makedirs(f'./{result_path}/explainable', exist_ok=True)
size = 256 # Original used 256
input_nc = 1
output_nc = 3

threshold_A = 60
threshold_B = 20
grad_layer = -1

NUM_WORKER =0
def display_diff(img_A, img_B, name):

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
def display_image_test(images, save_path):
    if images.shape[0] == 1:
        imgs = images.squeeze(dim=0)
    else: 
        imgs = images.permute(1,2,0)
    imgs_np = tensor2image(imgs)
    imgs_im = Image.fromarray(imgs_np)
    imgs_im.save(save_path)
def tensor2image(tensor):
    image = 127.5*(tensor.cpu().detach().float().numpy() + 1.0)
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
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG_A2B = torch.nn.DataParallel(netG_A2B).cuda()
    else:
        netG_A2B = netG_A2B.cuda()
    
    netG_A2B.load_state_dict(torch.load('./checkpoint/checkpointG_A2B_HE_XCG.pt'))
    transforms_test = [ transforms.ToPILImage(),
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5)) 
                    ]
    dataset_LR = data_dir_train  + '/DL_HR_Test_L'    
    n_list_train_LR = sorted(glob.glob('%s/*.png'%(dataset_LR)))
    print("Number of LR Samples in Test ",len(n_list_train_LR))      
    Test_dataloader = DataLoader(TestDataset(dataset_LR,transforms_PA=transforms_test), 
                     batch_size=1, shuffle=False, num_workers=NUM_WORKER)  
    for i, batch in enumerate(Test_dataloader):
        # Set model input
        real_A = Variable(batch["LR"].type(Tensor)) 
        name = batch["PA_name"]
        fake_B = netG_A2B(real_A) 

        real_A_mean = torch.mean(real_A,dim=1,keepdim=True)
        fake_B_mean = torch.mean(fake_B,dim=1,keepdim=True)
        real_A_normal = (real_A_mean - (threshold_A/127.5-1))*100
        fake_B_normal = (fake_B_mean - (threshold_A/127.5-1))*100
        real_A_sigmoid = torch.sigmoid(real_A_normal)
        fake_B_sigmoid = torch.sigmoid(fake_B_normal)

        print(name[0])
        display_image_test(fake_B[0], f"{result_path}/{name[0]}.png")  
        display_diff(real_A_sigmoid[0,0,...].detach().cpu(), fake_B_sigmoid[0,0,...].detach().cpu(), f"{result_path}/explainable/{name[0]}_diff.png")
        display_image_test(real_A_sigmoid[0], f"{result_path}/explainable/{name[0]}_A.png")  
        display_image_test(fake_B_sigmoid[0], f"{result_path}/explainable/{name[0]}_B.png")  
if __name__ == '__main__':
    main()    
