print('running ...')

import torch
import os
# import cv2
import argparse
# from KittiDataset import KittiDataset
from KittiAnchors import Anchors
import numpy
from torchvision import transforms

import numpy.typing as npt
from typing import cast, Type, Sequence
import typing

# RGB: typing.TypeAlias = 'numpy.dtype[numpy.uint8]'
# ThreeD: typing.TypeAlias = tuple[int, int, int]
# NDArrayRGB: typing.TypeAlias = 'numpy.ndarray[ThreeD, RGB]'

save_ROIs = True
max_ROIs = -1

def strip_ROIs(class_ID, label_list):
    ROIs = []
    for i in range(len(label_list)):
        ROI = label_list[i]
        if ROI[1] == class_ID:
            pt1 = (int(ROI[3]),int(ROI[2]))
            pt2 = (int(ROI[5]), int(ROI[4]))
            ROIs += [(pt1,pt2)]
    return ROIs

def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calc_max_IoU(ROI, ROI_list):
    max_IoU = 0
    for i in range(len(ROI_list)):
        max_IoU = max(max_IoU, calc_IoU(ROI, ROI_list[i]))
    return max_IoU

def main():

    print('running KittiToYoda ...')

    label_file = 'labels.txt'

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')
    argParser.add_argument('-IoU', metavar='IoU_threshold', type=float, help='[0.02]')
    argParser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    argParser.add_argument('-m', metavar='mode', type=str, help='[train/test]')
    argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')

    args = argParser.parse_args()

    input_dir = None
    if args.i != None:
        input_dir = args.i

    output_dir = None
    if args.o != None:
        output_dir = args.o

    IoU_threshold = 0.02
    if args.IoU != None:
        IoU_threshold = float(args.IoU)

    show_images = False
    if args.d != None:
        if args.d == 'y' or args.d == 'Y':
            show_images = True

    training = True
    if args.m == 'test':
        training = False

    use_cuda = False
    if args.cuda != None:
        if args.cuda == 'y' or args.cuda == 'Y':
            use_cuda = True

    labels = []

    device = 'cpu'
    if use_cuda == True and torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    dataset = KittiDataset(input_dir, training=training)
    anchors = Anchors()

    i = 0
    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        # print(i, idx, label)

        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        # print(car_ROIs)
        # for idx in range(len(car_ROIs)):
            # print(ROIs[idx])

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        if show_images:
            image1 = image.copy()
            for j in range(len(anchor_centers)):
                x = anchor_centers[j][1]
                y = anchor_centers[j][0]
                cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))
#############################
        # for j in range(len(centers)):
        #     center = centers[j]
        #     for k in range(len(anchor_shapes)):
        #         anchor_shape = anchor_shapes[k]
        #         pt1 = (int(center[0]-anchor_shape[0]/2), int(center[1]-anchor_shape[1]/2))
        #         pt2 = (int(center[0]+anchor_shape[0]/2), int(center[1]+anchor_shape[1]/2))
        #         cv2.rectangle(image, pt1, pt2, (0, 255, 255))
        
        #     cv2.imshow('image', image)
        #     key = cv2.waitKey(0)
        #     if key == ord('x'):
        #         break
#############################
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        # print('break 555: ', boxes)

        ROI_IoUs = []
        for idx in range(len(ROIs)):
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

        # print(ROI_IoUs)

        
        for k in range(len(boxes)):
            filename = str(i) + '_' + str(k) + '.png'
            if save_ROIs == True:
                cv2.imwrite(os.path.join(output_dir,filename), ROIs[k])
            name_class = 0
            name = 'NoCar'
            if ROI_IoUs[k] >= IoU_threshold:
                name_class = 1
                name = 'Car'
            labels += [[filename, name_class, name]]


        if show_images:
            cv2.imshow('image', image1)
            
#######
        key = cv2.waitKey(0)
        if key == ord('x'):
            break
#######################

        if show_images:
            image2 = image1.copy()

            for k in range(len(boxes)):
                if ROI_IoUs[k] > IoU_threshold:
                    box = boxes[k]
                    pt1 = (box[0][1],box[0][0])
                    pt2 = (box[1][1],box[1][0])
                    cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))
                    
#######################
                print(ROI_IoUs[k])
                box = boxes[k]
                pt1 = (box[0][1],box[0][0])
                pt2 = (box[1][1],box[1][0])
                cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))
                cv2.imshow('boxes', image2)
                key = cv2.waitKey(0)
                if key == ord('x'):
                    break
##############################

        if show_images:
            cv2.imshow('boxes', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break
# unncommented below
        # for j in range(len(label)):
        #     name = label[j][0]
        #     name_class = label[j][1]
        #     minx = int(label[j][2])
        #     miny = int(label[j][3])
        #     maxx = int(label[j][4])
        #     maxy = int(label[j][5])
    
        #     roi = image[miny:maxy,minx:maxx]
        #     # roi = cv2.resize(roi, (width,height))
    
        #     if save_full_ROI == True:
        #         filename = str(i) + '.png'
        #         cv2.imwrite(os.path.join(output_dir,filename), roi)
        #         labels += [[filename, name_class, name]]
        #         # print(i, filename, name_class, name)
    
        #     dy = maxy - miny + 1
        #     dx = maxx - minx + 1
        #     if anchors.min_range[0] < dy and anchors.min_range[1] < dx:
        #         for k in range(len(anchors.shapes)):
        #             shape = anchors.shapes[k]
        #             dy = int(((maxy - miny)-shape[0])/2)
        #             dx = int(((maxx - minx)-shape[1])/2)
        #             miny2 = miny + dy
        #             maxy2 = maxy - dy
        #             minx2 = minx + dx
        #             maxx2 = maxx - dx
    
        #             # print('break 08: ', miny2, maxy2, minx2, maxx2)
        #             if dx > 0 and dy > 0 and miny2 < maxy2 and minx2 < maxx2:
        #                 roi = image[miny2:maxy2, minx2:maxx2]
        #                 # roi = cv2.resize(roi, (width,height))
        #                 filename = str(i) + '_' + str(k) + '.png'
        #                 cv2.imwrite(os.path.join(output_dir, filename), roi)
        #                 labels += [[filename, name_class, name]]
    
        #                 if show_images == True:
        #                     image1 = image.copy()
        #                     cv2.rectangle(image1, (minx,miny), (maxx, maxy), (0,0,255))
        #                     cv2.imshow('image', image1)
        #                     cv2.imshow('roi', roi)
    
        #                     key = cv2.waitKey(0)
        #                     if key == ord('x'):
        #                         break
################################
        i += 1
        print(i)

        if max_ROIs > 0 and i >= max_ROIs:
            break
    #
    # print(labels)
    #
    if save_ROIs == True:
        with open(os.path.join(output_dir, label_file), 'w') as f:
            for k in range(len(labels)):
                filename = labels[k][0]
                name_class = str(labels[k][1])
                name = labels[k][2]
                f.write(filename + ' ' + name_class + ' ' + name + '\n')
        f.close()


###################################################################



class Anchors():
    grid = (4, 12)
    min_range = (100,100)
    max_range = (376,710)
    shape = [(150,150)]



def calc_anchor_centers(image_shape, anchor_grid):
    dy = int(image_shape[0]/anchor_grid[0])
    dx = int(image_shape[1]/anchor_grid[1])

    centers = []
    for y_idx in range(anchor_grid[0]):
        for x_idx in range(anchor_grid[1]):
            center_y = int((y_idx+1)*dy - dy/2)
            center_x = int((x_idx+1)*dx - dx/2)
            centers += [(center_y, center_x)]

    return centers

def get_anchor_ROIs(image, anchor_centers, anchor_shapes):
    ROIs = []
    boxes = []

    for j in range(len(anchor_centers)):
        center = anchor_centers[j]

        for k in range(len(anchor_shapes)):
            anchor_shape = anchor_shapes[k]
            pt1 = [int(center[0] - (anchor_shape[0]/2)), int(center[1] - (anchor_shape[1]/2))]
            pt2 = [int(center[0] + (anchor_shape[0]/2)), int(center[1] + (anchor_shape[1]/2))]

            # pt1 = [max(0, pt1[0]), min(pt1[1], image.shape[1])]
            # pt2 = [max(0, pt2[0]), min(pt2[1], image.shape[1])]
            pt1 = [max(0, pt1[0]), max(0, pt1[1])]
            pt2 = [min(pt2[0],  image.size[0]), min(pt2[1], image.size[1])]

            # print('break 777: ', pt1, pt2)
            ROI = [image.getbbox()]
            #ROI = [image.getpixel((pt1[0],pt2[0])), image.getpixel((pt1[1],pt2[1]))]
            ROIs += [ROI]
            boxes += [(pt1,pt2)]

    return ROIs, boxes

def batch_ROIs(ROIs, shape):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(shape)
    ])

    batch = torch.empty(size=(len(ROIs),3,shape[0],shape[1]))
    # batch = torch.empty(size=(shape[0],shape[1],len(ROIs)))
    # print('break 55: ', batch.shape)
    # print('break 55.5: ', shape)
    # resize = torchvision.transforms.Resize(size=shape)
    for i in range(len(ROIs)):
        ROI = ROIs[i]
        # print('break 650: ', ROI.shape, ROI.dtype)
        # print(ROI)
        # ROI = torchvision.transforms.ToTensor(ROI)
        # ROI = torch.from_numpy(ROI)
        # print('break 56: ', ROI.shape, ROI.dtype)
        # ROI = resize(ROI)
        ROI = transform(ROI)
        ROI = torch.swapaxes(ROI,1,2)
        # print('break 57: ', ROI.shape)
        # batch[i,:,:] = ROI[:,:]
        # batch = torch.cat([batch, ROI], dim=0)
        # print('break 664: ', i, batch.shape, ROI.shape)
        batch[i] = ROI
        # print('break 665: ', i, batch.shape)
    return batch

def minibatch_ROIs(ROIs, boxes, shape, minibatch_size):
    minibatch = []
    minibatch_boxes = []
    min_idx = 0
    while min_idx < len(ROIs)-1:
        max_idx = min(min_idx + minibatch_size, len(ROIs))
        minibatch += [batch_ROIs(ROIs[min_idx:max_idx], shape)]
        minibatch_boxes += [boxes[min_idx:max_idx]]
        min_idx = max_idx + 1
    return minibatch, minibatch_boxes

def strip_ROIs(class_ID, label_list):
    ROIs = []
    for i in range(len(label_list)):
        ROI = label_list[i]
        if ROI[1] == class_ID:
            pt1 = (int(ROI[3]),int(ROI[2]))
            pt2 = (int(ROI[5]), int(ROI[4]))
            ROIs += [(pt1,pt2)]
    return ROIs

def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calc_max_IoU(ROI, ROI_list):
    max_IoU = 0
    for i in range(len(ROI_list)):
        max_IoU = max(max_IoU, calc_IoU(ROI, ROI_list[i]))
    return max_IoU

from PIL import Image
def carBox(image, label=1):
    image = Image.open(image)
    if label != 1:
        return None
    
    anchors = Anchors()
    centers = calc_anchor_centers(image.size, anchors.grid) 
    ROIs, boxes = get_anchor_ROIs(image, centers, anchors.shape)
    print(ROIs, boxes)
    
    batchROIs = batch_ROIs(ROIs, anchors.shape)
    print(batchROIs)
    
    # ROI_IoUs = []
    # for idx in range(len(ROIs)):
    #     ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]
    
carBox('./data/Kitti8_ROIs/8_19.png')

