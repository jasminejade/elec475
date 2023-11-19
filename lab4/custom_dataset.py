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
        self.dir = "D:/475/elec475/lab4/data/Kitti8_ROIs/train/"
        self.images = ''
        self.png = []
        self.setLabels()

    def setLabels(self, filepath='D:/475/elec475/lab4/data/Kitti8_ROIs/train/labels.txt'):
        images = {}
        with open(filepath, 'r') as f:
            labels = f.readlines()
        f.close()
        i = 0
        for x in labels:
            x = x.strip('\n')
            self.png.append(x.split(' ')[0])
            image = f'{self.dir}' + x.split(' ')[0]
            value = x.split(' ')[1]
            label = x.split(' ')[2]

            images[i] = [image, float(value), label]
            i += 1
        self.images = images
        # img = Image.open(filepath + image_path).convert("RGB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index][0]).convert('RGB')

        # try:
        #     image = Image.open(self.images[index][0]).convert("RGB")
        # except Exception as e:
        #     print(index, e)
        #     return None

        # image_sample = self.transform(image)
        # print('break 27: ', index, image, image_sample.shape)

        if self.transform:
            image = self.transform(image)
        # print(self.images[2883][0])
        # print(len(self.images))

        return image, int(self.images[index][1])

blessed = custom_dataset('D:/475/elec475/lab4/data/Kitti8_ROIs/train/')
blessed.setLabels()
blessed.__getitem__(0)