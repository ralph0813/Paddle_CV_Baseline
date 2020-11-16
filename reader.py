import cv2
import numpy as np
import random


def transform_img(img, augment=True):
    # 将图片尺寸缩放道 224x224
    diff = int((img.shape[0] - img.shape[1]) / 2)
    if diff > 0:
        img = cv2.copyMakeBorder(img, 0, 0, diff, diff, cv2.BORDER_WRAP)
    else:
        img = cv2.copyMakeBorder(img, -diff, -diff, 0, 0, cv2.BORDER_WRAP)
    img = cv2.resize(img, (224, 224))
    # # 数据增强
    # if augment == True:
    #     # img = image_augment(img)
    #     img = image_augment_imgauglib(img)

    # show pic
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # # print(img.shape)

    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    # img = np.mean(img, axis = 0).reshape((1, 28, 28))
    return img


def tra_loader(dataDir="data/", batch_size=8):
    # 将datadir目录下的文件列出来，每条文件都要读入
    fileNames = np.loadtxt(dataDir + "labels/Train.txt", dtype=np.str)
    np.random.shuffle(fileNames)

    def reader():
        batch_imgs = []
        batch_labels = []
        for name in fileNames:
            img = cv2.imread(dataDir + "Images/" + name[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform_img(img)
            label = name[1]
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []
        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


def val_loader(dataDir="data/", batch_size=8):
    # 将datadir目录下的文件列出来，每条文件都要读入
    fileNames = np.loadtxt(dataDir + "labels/Eval.txt", dtype=np.str)

    def reader():
        batch_imgs = []
        batch_labels = []
        for name in fileNames:
            img = cv2.imread(dataDir + "Images/" + name[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform_img(img, False)
            label = name[1]
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []
        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


def test_loader(dataDir="data/", batch_size=8):
    # 将datadir目录下的文件列出来，每条文件都要读入
    fileNames = np.loadtxt(dataDir + "labels/Test.txt", dtype=np.str)

    def reader():
        batch_imgs = []
        for name in fileNames:
            img = cv2.imread(dataDir + "Images/" + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform_img(img, augment=False)
            batch_imgs.append(img)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                yield imgs_array
                batch_imgs = []
        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            yield imgs_array

    return reader
