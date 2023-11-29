
def getBoxes(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    anchors = Anchors()
    centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
    ROIs, boxes = anchors.get_anchor_ROIs(image, centers, anchors.shapes)
    return image, boxes

def test2(network, test_loader, device='cuda', box_coor=None):

    network.to(device=device)
    network.eval()

    total = 0
    top5total = 0
    boxes = np.zeros(48)
    #print(box_coor)
    with torch.no_grad():
        imgNum = 0
        for j, (imgs, labels) in enumerate(test_loader):
            print(j)
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            output = network(imgs)  # forward method

            predict = torch.round(torch.sigmoid(output))  # get index of class with highest probability

            if imgNum < 10:
                img_pth = './data/Kitti8/test/image/00600' + str(j) + '.png'
            elif imgNum < 100 and imgNum >= 10:
                img_pth = './data/Kitti8/test/image/0060' + str(j) + '.png'
            else:
                img_pth = './data/Kitti8/test/image/006' + str(j) + '.png'
            image, boxes1 = getBoxes(img_pth)
            image2 = image.copy()
            print(len(predict), len(boxes1))
            classified = 0
            index =0

            for i in range(0, len(predict)):
                if predict[i] == 1:
                    boxes[i] = 1
                    index = i
                    box = boxes1[i]
                    pt1 = (box[0][1], box[0][0])
                    pt2 = (box[1][1], box[1][0])
                    print(pt1, pt2)
                    cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))
                    # cv2.imshow('boxes', image2)
                if predict[i] == labels[i]:
                    classified += 1
                # image2 = image2.cpu()
                # image2 = np.asarray(imgs[i])
                # image2 = np.transpose(image2, (1,2,0))

                # for k in range(len(boxes1)):
                #     if boxes[k] == 1:
                #         box = boxes1[k]
                #         pt1 = (box[0][1],box[0][0])
                #         pt2 = (box[1][1],box[1][0])
                #         print(pt1, pt2)
                #         cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))
                #cv2.imshow('boxes', image2)
            cv2.imshow('boxes', image2)
            imgNum +=1
            #     image3 = cv2.imread(img_pth, cv2.IMREAD_COLOR)
            #    # image2 = Image.open(img_pth).convert('RGB')
            #     #draw = ImageDraw.Draw(image2)
            #     for j in range(0,len(boxes)-1,2):
            #         print(box_coor[j])
            #         if boxes[j] == 1:
            #             pt1 = (box_coor[j,0], box_coor[j,1])
            #             pt2 = (box_coor[j+1,0], box_coor[j+1,1])
            #             print(pt1, pt2)
            #            # draw.rectangle(pt1, pt2)
            #             cv2.rectangle(image3, box_coor[j], box_coor[j+1], color=(0, 255, 255))

            #         cv2.imshow('boxes', image3)

            #         key = cv2.waitKey(0)
            #         if key == ord('x'):
            #             break
                #image2.show()
                # cv2.imshow('boxes', image3)
                # total += classified
                # print(f'classified: {classified}/{len(predict)}')

    # print(f'total classified: {total}/{len(test_set)}')
    # print(f'% Accuracy: {total / len(test_set) * 100}%')