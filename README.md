# XDL_MIR_PAM_2024
## System Requirements
You need Pytorch_with_CUDA for this experiments (pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04)
And following additional packages are neede
-    matplotlib==3.7.5
-    pytorch-msssim==1.0.0
-    opencv-python==4.10.0.84
-    captum==0.7.0

## Installtion guide
Set environment with following:
`sh requirements.sh`

## Data Preparation
Train dataset should be positioned in 'Data' Folder.
- Data/Train_LR_256 # Low-resolution data in png format, cropped into patches of 256x256 size.
- Data/Train_gHR_256 # High-resolution gray data in png format, cropped into patches of 256x256 size.
- Data/Train_HR_256 # High-resolution color data in png format, cropped into patches of 256x256 size.
Test dataset shuld be positioned in 'test' Folder
- test/Test_L # Low-resolution test data in png format, cropped into patches of 256x256 size.
- test/DL_HR_Test_L # High-resolution test color data in png format, cropped into patches of 256x256 size.(Typically use the test result data from Step 1)

## Demo Introduction
### Step 0. Checkpoint Preparation  
Download the checkpoint parameters from [link](https://1drv.ms/f/c/de011cb09ae2716d/EiGgV_zRc1pJuomYOIJWEpsBt7JAHsZ8kYIIUGZD_mlPeQ?e=UR11ty) and place them in the 'checkpoint' folder.  
### Step 1. Test LR to HR  
Test low resolution to high resolution transform system with following code.  
'python test_XCG.py'  
Expected output: test/DL_HR_Test_L/{NAME}.png (3 mins for test)  
Explainable results:  
1. test/DL_HR_Test_L/explainable/{NAME}_A.png : Saliency mask of input  
2. test/DL_HR_Test_L/explainable/{NAME}_B.png : Saliency mask of generated output  
3. test/DL_HR_Test_L/explainable/{NAME}_diff.png : Saliency mask difference between input and generated output  
4. test/DL_HR_Test_L/explainable/{NAME}_grad_{grad_layer}.png : gradCAM output of layer {grad_layer}  

### Step 2. Test gray HR to Color HR  
Test gray to color transform system with following code.  
'python test_XCG_HE.py'  
Expected output: test/DL_HE_Test_H/{NAME}.png (3 mins for test)  
Explainable results: 
1. test/DL_HE_Test_H/explainable/{NAME}_A.png : Saliency mask of input  
2. test/DL_HE_Test_H/explainable/{NAME}_B.png : Saliency mask of generated output  
3. test/DL_HE_Test_H/explainable/{NAME}_diff.png : Saliency mask difference between input and generated output  
