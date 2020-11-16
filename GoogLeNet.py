import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Dropout, BatchNorm

from train import train


class Inception(fluid.dygraph.Layer):
    def __init__(self, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1_1 = Conv2D(c1[0], c1[1], 1, act='relu')
        self.p2_1 = Conv2D(c1[0], c2[0], 1, act='relu')
        self.p2_2 = Conv2D(c2[0], c2[1], 3, padding=1, act='relu')
        self.p3_1 = Conv2D(c1[0], c3[0], 1, act='relu')
        self.p3_2 = Conv2D(c3[0], c3[1], 5, padding=2, act='relu')
        self.p4_1 = Pool2D(pool_size=3, pool_stride=1, pool_padding=1, pool_type='max')
        self.p4_2 = Conv2D(c1[0], c4, 1, act='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return fluid.layers.concat([p1, p2, p3, p4], axis=1)


class GoogLeNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(GoogLeNet, self).__init__()

        self.conv1 = Conv2D(3, 64, 7, padding=3, stride=2, act='relu')
        self.pool1 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.conv2_1 = Conv2D(64, 64, 1, act='relu')
        self.conv2_2 = Conv2D(64, 192, 3, padding=1, act='relu')
        self.pool2 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.block3_a = Inception((192, 64), (96, 128), (16, 32), 32)
        self.block3_b = Inception((256, 128), (128, 192), (32, 96), 64)
        self.pool3 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.block4_a = Inception((480, 192), (96, 208), (16, 48), 64)
        self.block4_b = Inception((512, 160), (112, 224), (24, 64), 64)
        self.block4_c = Inception((512, 128), (128, 256), (24, 64), 64)
        self.block4_d = Inception((512, 112), (144, 288), (32, 64), 64)
        self.block4_e = Inception((528, 256), (160, 320), (32, 128), 128)
        self.pool4 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.block5_a = Inception((832, 256), (160, 320), (32, 128), 128)
        self.block5_b = Inception((832, 384), (192, 384), (48, 128), 128)
        self.pool5 = Pool2D(pool_size=7, pool_stride=1, global_pooling=True, pool_type='avg')
        self.drop = Dropout(p=0.4)
        self.fc = Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.block3_b(self.block3_a(x)))
        x = self.pool4(self.block4_e(self.block4_d(self.block4_c(self.block4_b(self.block4_a(x))))))
        x = self.pool5(self.block5_b(self.block5_a(x)))
        x = self.drop(x)
        x = fluid.layers.reshape(x, [-1, 1024])
        x = self.fc(x)

        return x


class GoogLeNet_BN(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(GoogLeNet_BN, self).__init__()

        self.bn64 = BatchNorm(64)
        self.bn192 = BatchNorm(192)
        self.bn256 = BatchNorm(256)
        self.bn480 = BatchNorm(480)
        self.bn512 = BatchNorm(512)
        self.bn528 = BatchNorm(528)
        self.bn832 = BatchNorm(832)
        self.bn1024 = BatchNorm(1024)

        self.conv1 = Conv2D(3, 64, 7, padding=3, stride=2, act='relu')
        self.pool1 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.conv2_1 = Conv2D(64, 64, 1, act='relu')
        self.conv2_2 = Conv2D(64, 192, 3, padding=1, act='relu')
        self.pool2 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.block3_a = Inception((192, 64), (96, 128), (16, 32), 32)
        self.block3_b = Inception((256, 128), (128, 192), (32, 96), 64)
        self.pool3 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.block4_a = Inception((480, 192), (96, 208), (16, 48), 64)
        self.block4_b = Inception((512, 160), (112, 224), (24, 64), 64)
        self.block4_c = Inception((512, 128), (128, 256), (24, 64), 64)
        self.block4_d = Inception((512, 112), (144, 288), (32, 64), 64)
        self.block4_e = Inception((528, 256), (160, 320), (32, 128), 128)
        self.pool4 = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.block5_a = Inception((832, 256), (160, 320), (32, 128), 128)
        self.block5_b = Inception((832, 384), (192, 384), (48, 128), 128)
        self.pool5 = Pool2D(pool_size=7, pool_stride=1, global_pooling=True, pool_type='avg')
        self.drop = Dropout(p=0.4)
        self.fc = Linear(1024, num_classes)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.bn64(x)
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        # x = self.bn192(x)
        x = self.pool3(self.block3_b(self.block3_a(x)))
        x = self.bn480(x)
        x = self.pool4(self.block4_e(self.block4_d(self.block4_c(self.block4_b(self.block4_a(x))))))
        # x = self.bn832(x)
        x = self.pool5(self.block5_b(self.block5_a(x)))
        x = self.bn1024(x)
        x = self.drop(x)
        x = fluid.layers.reshape(x, [-1, 1024])
        x = self.fc(x)

        return x


def train_model():
    epoch_num = 60
    batch_size = 32
    with fluid.dygraph.guard():
        model = GoogLeNet(num_classes=16)
        train(epoch_num, batch_size, model)


if __name__ == '__main__':
    train_model()
