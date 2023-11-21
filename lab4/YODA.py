import torch
import cv2
from torch import nn
from torchvision import transforms
from KittiDataset import KittiDataset
from KittiAnchors import *
import numpy as np
from matplotlib import pyplot as plt

class YODA(nn.Module):
    show = False
    IoU_threshold = 0.02
    def __init__(self, classifier_pth=None):
        super().__init__()
        self.classifer = classifier_pth
        self.image = []
        self.labels = []
        self.ROIs = []
        self.anchors = Anchors()
        self.boxes = []
        
        self.class_label = {'DontCare': 0, 'Misc': 1, 'Car': 2, 'Truck': 3, 'Van': 4, 'Tram': 5, 'Cyclist': 6, 'Pedestrian': 7,
                   'Person_sitting': 8}
        

        
    def subdivide(self, image_path, label_path):
        
        # subdivide single image into set of ROIs
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        class_ID = self.class_label['Car']
        car_ROIs = []
        
        labels_string = None

   
        with open(label_path) as label_file:
            labels_string = label_file.readlines()

        for i in range(len(labels_string)):
            lsplit = labels_string[i].split(' ')
            label = [lsplit[0], int(self.class_label[lsplit[0]]), float(lsplit[4]), float(lsplit[5]), float(lsplit[6]), float(lsplit[7])]
            self.labels += [label]
            
        # KittiDataset.strip_ROIs
        for i in range(len(self.labels)):
            ROI = self.labels[i]
            if ROI[1] == class_ID:
                pt1 = (int(ROI[3]),int(ROI[2]))
                pt2 = (int(ROI[5]), int(ROI[4]))
                car_ROIs += [(pt1,pt2)]

        centers = self.anchors.calc_anchor_centers(self.image.shape, self.anchors.grid)
        self.ROIs, self.boxes = self.anchors.get_anchor_ROIs(self.image, centers, self.anchors.shapes)
        
        # KittiAnchors get_anchots_ROIs
        # for j in range(len(centers)):
        #     center = centers[j]
        #     for k in range(len(self.anchors.shapes)):
        #         anchor_shape = self.anchors.shapes[k]
        #         pt1 = (int(center[0]-anchor_shape[0]/2), int(center[1]-anchor_shape[1]/2))
        #         pt2 = (int(center[0]+anchor_shape[0]/2), int(center[1]+anchor_shape[1]/2))

        #         cv2.rectangle(self.image, pt1, pt2, (0, 255, 255))
                
        #         ROI = self.image[pt1[0]:pt2[0],pt1[1]:pt2[1],:]
        #         self.ROIs += [ROI]
        #         self.boxes += [(pt1,pt2)]
        
        #     if self.show:
        #         cv2.imshow('image', self.image)
        #         key = cv2.waitKey(0)
        #         if key == ord('x'):
        #             break
    
        ROI_IoUs = []
        for idx in range(len(self.ROIs)):
            ROI_IoUs += [self.anchors.calc_max_IoU(self.boxes[idx], car_ROIs)]
        
        image2 = self.image.copy()
        # for k in range(len(self.boxes)):
        #     # if ROI_IoUs[k] > self.IoU_threshold:
        #     #     box = self.boxes[k]
        #     #     pt1 = (box[0][1],box[0][0])
        #     #     pt2 = (box[1][1],box[1][0])
        #     #     cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

        #     print(ROI_IoUs[k])
        #     box = self.boxes[k]
        #     pt1 = (box[0][1],box[0][0])
        #     pt2 = (box[1][1],box[1][0])
        #     cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))
        #     cv2.imshow('boxes', image2)
        #     key = cv2.waitKey(0)
        #     if key == ord('x'):
        #         break
        for j in range(len(self.boxes)):
            coor1 = self.boxes[j][0]
            coor2 = self.boxes[j][1]
            minx = int(coor1[0])
            miny = int(coor1[1])
            maxx = int(coor2[0])
            maxy = int(coor2[1])
            roi = self.image[miny:maxy,minx:maxx]
            print(roi)
            for k in range(len(self.anchors.shapes)):
                shape = self.anchors.shapes[k]
                dy = int(((maxy - miny)-shape[0])/2)
                dx = int(((maxx - minx)-shape[1])/2)
                miny2 = miny + dy
                maxy2 = maxy - dy
                minx2 = minx + dx
                maxx2 = maxx - dx
                
                # cv2.rectangle(roi, (minx2,miny2), (maxx2, maxy2), (0,0,255))
                # cv2.imshow('image', image2)
                # cv2.imshow('roi', roi)
    
                # key = cv2.waitKey(0)
                # if key == ord('x'):
                #     break

    def batch_ROIs(self, ROIs, shape):
        print(shape)
        print(len(ROIs))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(shape[1], shape[0]))
        ])

        batch = torch.empty(size=(len(ROIs),3, shape[0],shape[1]))
        #cbatch = torch.empty(size=(shape[0],shape[1],len(ROIs)))
        # print('break 55: ', batch.shape)
        # print('break 55.5: ', shape)
        #resize = transforms.Resize(size=(shape[0], shape[1]))
        for i in range(len(ROIs)):
            ROI = ROIs[i]
            
            # print('break 650: ', ROI.shape, ROI.dtype)
           # ROI = np.asarray(ROI)
            #ROI = torch.from_numpy(ROI)
            ROI = transform(ROI)
            # print('break 56: ', ROI.shape, ROI.dtype)
            #ROI = resize(ROI)
            #ROI = transforms.ToTensor(ROI)
            ROI = torch.swapaxes(ROI,1,2)
            # print('break 57: ', ROI.shape)
            # batch[i,:,:] = ROI[:,:]
            # batch = torch.cat([batch, ROI], dim=0)
            # print('break 664: ', i, batch.shape, ROI.shape)
            batch[i] = ROI
            # print('break 665: ', i, batch.shape)
        return batch

    def minibatch_ROIs(self, ROIs, boxes, shape, minibatch_size):
        minibatch = []
        minibatch_boxes = []
        min_idx = 0
        while min_idx < len(ROIs)-1:
            max_idx = min(min_idx + minibatch_size, len(ROIs))
            minibatch += [self.batch_ROIs(ROIs[min_idx:max_idx], shape)]
            minibatch_boxes += [boxes[min_idx:max_idx]]
            min_idx = max_idx + 1
        return minibatch, minibatch_boxes


    def buildBatch(self):
        batch = self.batch_ROIs(self.ROIs, self.image.shape) # this is a tensor of tensors

        minibatch, minibatch_boxes = self.minibatch_ROIs(self.ROIs, self.boxes, self.image.shape, 48)
        #print(minibatch,'\n', minibatch_boxes)
        
        for b in minibatch:
            plt.imshow(b)
            
        plt.show()
            
        
    def display(self, batchROIs):
        return None
                
        #         ROI = self.image[pt1[0]:pt2[0],pt1[1]:pt2[1],:]
        #         self.ROIs += [ROI]
        #         self.boxes += [(pt1,pt2)]
        
        #     if self.show:
        #         cv2.imshow('image', self.image)
        #         key = cv2.waitKey(0)
        #         if key == ord('x'):
        #             break
        
model = YODA()
model.subdivide('./data/Kitti8/000008.png', './data/Kitti8/000008.txt')
model.buildBatch()