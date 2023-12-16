# elec 475 custom_dataset.py
#
# assigns label to images
#

import os

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

class custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # self.transform = transforms.Compose([
        #     transforms.Resize(size=(size,size), interpolation=Image.BICUBIC),
        #     # transforms.RandomCrop(crop_size),
        #     transforms.ToTensor()
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        self.transform = transform
        # os.chdir('..')
        #self.image_files = [dir + file_name for file_name in os.listdir(dir)]
        #print(os.getcwd())
        self.dir = dir # images directory

        self.images = {}

    def setLabels(self, filepath):
        images = {}
        with open(filepath, 'r') as f:
            labels = f.readlines()
        f.close()
        i = 0
        for x in labels:
            x = x.strip(' ')
            pet = x.strip('\n').split(',', 1)
            xy = pet[1].strip("\"(").strip(")\"").split(',')
            #print(xy)
            images[i] = [f'{self.dir}{pet[0]}', xy]
            i += 1
        self.images = images


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x_coord = int(self.images[index][1][0])
        y_coord = int(self.images[index][1][1])


        coords = torch.tensor([x_coord, y_coord], dtype=torch.float32)


        # Apply the scaling factor to the coordinates
        # scaled_coords = coords * scale_factor


        image = Image.open(self.images[index][0]).convert('RGB')

        size = image.size
        coords[0] = coords[0] * (224 / size[0])
        coords[1] = coords[1] * (224 / size[1])

        # coords[0] = coords[0] * (224 / size[0])
        # coords[1] = coords[1] * (224 / size[1])

        if self.transform:
            image = self.transform(image)
        #print(self.images[1])
        return image, coords
       # xy = [int(self.images[index][1][0]), int(self.images[index][1][1])]

        #return image, torch.tensor[int(self.images[index][1][0]), int(self.images[index][1][1])]

# os.chdir('..')
# folder = './oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images/'
# file = './oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.2.txt'
# blessed = custom_dataset(folder)
# blessed.setLabels(file)
# image, nose = blessed.__getitem__(0)
# print(nose.x, nose.y)