import glob
import os
import torchvision.transforms as transforms
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFilter
class ImageDataset(Dataset):
    def __init__(self, root_PA,root_HE, transforms_PA=None, transforms_HE=None, train=True, unaligned=True):
        self.transform_PA = transforms.Compose(transforms_PA)
        self.transform_HE = transforms.Compose(transforms_HE)
        self.files_PA = []
        self.files_HE = []
        self.files_PA = sorted(glob.glob(os.path.join(root_PA)+ '/*.png'))
        self.files_HE = sorted(glob.glob(os.path.join(root_HE)+ '/*.png'))

    def __getitem__(self, index):
        PA_npdata = Image.open(self.files_PA[index % len(self.files_PA)])
        #PA_npdata = PA_npdata.filter(ImageFilter.SHARPEN)
        HE_npdata = Image.open(self.files_HE[index % len(self.files_HE)])
        img2 = np.zeros( ( np.array(PA_npdata).shape[0], np.array(PA_npdata).shape[1], 1 ) )
        img2[:,:,0] = PA_npdata
        PA_npdata = np.transpose(img2,(2,0,1))
        PA_npdata = torch.squeeze(torch.Tensor(PA_npdata/255.0)) 
        item_PA = self.transform_PA(PA_npdata)

        img3 = np.zeros( ( np.array(HE_npdata).shape[0], np.array(HE_npdata).shape[1], 1 ) )
        img3[:,:,0] = HE_npdata
        HE_npdata = np.transpose(img3,(2,0,1))
        HE_tsdata_ch3 = torch.squeeze(torch.Tensor(HE_npdata/255.0))
        item_HE = self.transform_HE(HE_tsdata_ch3)

        return {'PA' : item_PA, 'HE' : item_HE, 'PA_class' : 0, 'HE_class' : 0 }
        
    def __len__(self):
        return len(self.files_PA)
        
    
    
class TestDataset(Dataset):
    def __init__(self, root_PA, transforms_PA=None):
        self.transform_PA = transforms.Compose(transforms_PA)

        self.files_PA = []
       
        self.files_PA = sorted(glob.glob(os.path.join(root_PA) + '/*.png'))

    def __getitem__(self, index):
        PA_npdata = Image.open(self.files_PA[index % len(self.files_PA)]) 
        PA_npdata = PA_npdata.filter(ImageFilter.SHARPEN)
        name = Path(self.files_PA[index]).stem
        img2 = np.zeros( ( np.array(PA_npdata).shape[0], np.array(PA_npdata).shape[1], 1 ) )
        img2[:,:,0] = PA_npdata
        PA_npdata = np.transpose(img2,(2,0,1))
        PA_npdata = torch.squeeze(torch.Tensor(PA_npdata/255.0)) # data reverse
        item_PA = self.transform_PA(PA_npdata)
        return {'LR' : item_PA, 'PA_name' : name, 'LR_class' : 0}
        
    def __len__(self):
        return len(self.files_PA)     