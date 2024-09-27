# XDL_MIR_PAM_2024


### System Requirements
You need Pytorch_with_CUDA for this experiments (pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04)
And following additional packages are neede
-    matplotlib==3.7.5
-    pytorch-msssim==1.0.0

### Installtion guide
Set up your environment with Codeocean system.
Setting up on a typical desktop takes about a minute.

### Data Preparation
Train dataset should be positioned in 'Data' Folder.
- Data/Train_LR_256 # Low-resolution data in png format, cropped into patches of 256x256 size.
- Data/Train_gHR_256 # High-resolution gray data in png format, cropped into patches of 256x256 size.
- Data/Train_HR_256 # High-resolution color data in png format, cropped into patches of 256x256 size.
Test dataset shuld be positioned in 'test' Folder
- test/Test_L # Low-resolution test data in png format, cropped into patches of 256x256 size.
- test/Test_H # High-resolution test color data in png format, cropped into patches of 256x256 size.(Typically use the test result data from Step 1)

### Demo Introduction
- Step 1. Train LR to HR
Train lowresolution to highresolution transform system with following code.
'python train_XCG.py'
Expected output: checkpoint/checkpointG_A2B_HR_XCG.pt (2 days for training)

- Step 2. Test LR to HR
Test lowresolution to highresolution transform system with following code.
'python test_XCG.py'
Expected output: test/DL_HR_Test_L/DL_{NAME}.png (3 mins for test)

- Step 3. Train gray HR to Color HR
Train gray to color transform system with following code.
'python train_XCG_HE.py'
Expected output: checkpoint/checkpointG_A2B_HE_XCG.pt (2 days for training)

- Step 4. Test gray HR to Color HR
Test gray to color transform system with following code.
'python test_XCG_HE.py'
Expected output: test/DL_HE_Test_H/DL_{NAME}.png (3 mins for test)
