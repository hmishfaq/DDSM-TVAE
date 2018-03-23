from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio
import torch
from scipy import misc
import torchvision as vision


filenames = {'train': ['triplet_train.csv'],
             'val': ['triplet_val.csv'],
             'test': ['triplet_test.csv']}



# image loader for training and testing with mean and std for pixel values
def default_image_loader(path):
    img = misc.imread(path)
    imgn = np.where(img>0,img,np.nan)
    mean = np.nanmean(imgn,axis=(0,1))
    std =np.nanstd(imgn,axis=(0,1))
    return img, mean, std 


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, base_path, filenames_filename, split, n_triplets, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.mat
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.root = root
        self.base_path = base_path  
        self.filenamelist = []
        print(os.path.join(self.root, filenames_filename))
        for line in open(os.path.join(self.root, filenames_filename)):
        
            self.filenamelist.append(line.rstrip('\n\x00'))
        triplets = []
        if split == 'train':
            fnames = filenames['train']
        elif split == 'val':
            fnames = filenames['val']
        else:
            fnames = filenames['test']

        for line in open(os.path.join(self.root, split, fnames[0])):
            triplets.append((line.split(',')[0], line.split(',')[1], line.split(',')[2])) # anchor, far, close   
        np.random.shuffle(triplets)
        self.triplets = triplets[:int(n_triplets)]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        
        if os.path.exists(os.path.join('..', self.filenamelist[int(path1)])) and os.path.exists(os.path.join('..', self.filenamelist[int(path2)])) and os.path.exists(os.path.join('..', self.filenamelist[int(path3)])):
           
            img1, mean1, std1 = self.loader(os.path.join('..', self.filenamelist[int(path1)]))
            img2, mean2, std2 = self.loader(os.path.join('..', self.filenamelist[int(path2)]))
            img3, mean3, std3  = self.loader(os.path.join('..', self.filenamelist[int(path3)]))

            #refer to PyTorch and PIL documentation for "transform" and "PIL" specific things
            toPIL = transforms.ToPILImage()
            img1 = toPIL(img1)
            img2 = toPIL(img2)
            img3 = toPIL(img3)
            
            normalize1 = transforms.Normalize(mean=mean1, std=std1)
            normalize2 = transforms.Normalize(mean=mean2, std=std2)
            normalize3 = transforms.Normalize(mean=mean3, std=std3)


            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)

            img1 = normalize1(img1)
            img2 = normalize2(img2)
            img3 = normalize3(img3)
            
            return img1, img2, img3
        else:
            return None

    def __len__(self):
        return len(self.triplets)
