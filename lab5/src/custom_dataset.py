# elec 475 custom_dataset.py
#
# assigns label to images
#

import os
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
        self.image_files = [dir + file_name for file_name in os.listdir(dir)]
        self.dir = dir # images directory
        self.images = ''

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

            images[i] = PetNose(f'{self.dir}{pet[0]}', xy[0], xy[1])
            i += 1
        self.images = images


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index].pet).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.images[index]

class PetNose(object):
    def __init__(self, pet, x, y):
        self.pet = pet
        self.x = x
        self.y = y


os.chdir('..')
folder = './oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images/'
file = './oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.2.txt'
blessed = custom_dataset(folder)
blessed.setLabels(file)
image, nose = blessed.__getitem__(0)
print(nose.x, nose.y)