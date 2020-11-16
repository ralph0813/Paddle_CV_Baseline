import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Dropout, BatchNorm

from train import train


# 搭建一个基本的CNN模型
class CNN(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()

        self.conv1 = Conv2D(3, 64, 5, padding=2, stride=1, act='sigmoid')
        self.conv2 = Conv2D(64, 128, 5, padding=2, stride=1, act='sigmoid')
        self.conv3 = Conv2D(128, 256, 5, padding=2, stride=1, act='sigmoid')
        self.conv4 = Conv2D(256, 512, 5, padding=2, stride=1, act='sigmoid')
        self.conv5 = Conv2D(512, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.conv6 = Conv2D(1024, 1024, 5, padding=2, stride=1, act='sigmoid')

        self.fc1 = Linear(1024 * 7 * 7, 1024, act='sigmoid')
        self.fc2 = Linear(1024, num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.pool_down(x)
        x = self.conv5(x)
        x = self.pool_down(x)
        x = self.conv6(x)

        x = fluid.layers.reshape(x, [-1, 1024 * 7 * 7])
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class CNN_BaseLine(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN_BaseLine, self).__init__()

        self.conv1 = Conv2D(3, 64, 5, padding=2, stride=1, act='sigmoid')
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 5, padding=2, stride=1, act='sigmoid')
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 5, padding=2, stride=1, act='sigmoid')
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 5, padding=2, stride=1, act='sigmoid')
        self.bn4 = BatchNorm(512)
        self.conv5 = Conv2D(512, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.bn5 = BatchNorm(1024)
        self.conv6 = Conv2D(1024, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.bn6 = BatchNorm(1024)

        self.fc1 = Linear(1024 * 7 * 7, 1024, act='sigmoid')
        self.fc2 = Linear(1024, num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool_down(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool_down(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = fluid.layers.reshape(x, [-1, 1024 * 7 * 7])
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class CNN_LeakyRelu(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN_LeakyRelu, self).__init__()

        self.conv1 = Conv2D(3, 64, 5, padding=2, stride=1, act='leaky_relu')
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 5, padding=2, stride=1, act='leaky_relu')
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 5, padding=2, stride=1, act='leaky_relu')
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 5, padding=2, stride=1, act='leaky_relu')
        self.bn4 = BatchNorm(512)
        self.conv5 = Conv2D(512, 1024, 5, padding=2, stride=1, act='leaky_relu')
        self.bn5 = BatchNorm(1024)
        self.conv6 = Conv2D(1024, 1024, 5, padding=2, stride=1, act='leaky_relu')
        self.bn6 = BatchNorm(1024)

        self.fc1 = Linear(1024 * 7 * 7, 1024, act='leaky_relu')
        self.fc2 = Linear(1024, num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool_down(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool_down(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = fluid.layers.reshape(x, [-1, 1024 * 7 * 7])
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class CNN_PoolReplaceFC(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN_PoolReplaceFC, self).__init__()

        self.conv1 = Conv2D(3, 64, 5, padding=2, stride=1, act='sigmoid')
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 5, padding=2, stride=1, act='sigmoid')
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 5, padding=2, stride=1, act='sigmoid')
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 5, padding=2, stride=1, act='sigmoid')
        self.bn4 = BatchNorm(512)
        self.conv5 = Conv2D(512, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.bn5 = BatchNorm(1024)
        self.conv6 = Conv2D(1024, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.bn6 = BatchNorm(1024)

        # 用此处的全局平均池化代替原来的全连接层
        self.pool_global = Pool2D(pool_stride=1, global_pooling=True, pool_type='avg')
        self.fc = Linear(input_dim=1024, output_dim=num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool_down(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool_down(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = self.pool_global(x)
        x = fluid.layers.reshape(x, [-1, 1024])

        x = self.fc(x)

        return x


class CNN_DropOut(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN_DropOut, self).__init__()

        self.conv1 = Conv2D(3, 64, 5, padding=2, stride=1, act='sigmoid')
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 5, padding=2, stride=1, act='sigmoid')
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 5, padding=2, stride=1, act='sigmoid')
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 5, padding=2, stride=1, act='sigmoid')
        self.bn4 = BatchNorm(512)
        self.conv5 = Conv2D(512, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.bn5 = BatchNorm(1024)
        self.conv6 = Conv2D(1024, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.bn6 = BatchNorm(1024)

        self.fc1 = Linear(1024 * 7 * 7, 1024, act='sigmoid')
        self.fc2 = Linear(1024, num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 按照50%的比例随机丢弃部分神经元
        self.dropout = Dropout(p=0.5)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool_down(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool_down(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = fluid.layers.reshape(x, [-1, 1024 * 7 * 7])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNN_CoreSize3(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN_CoreSize3, self).__init__()

        self.conv1 = Conv2D(3, 64, 3, padding=1, stride=1, act='sigmoid')
        self.bn1 = BatchNorm(64)
        self.conv1_2 = Conv2D(64, 64, 3, padding=1, stride=1, act='sigmoid')
        self.bn1_2 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 3, padding=1, stride=1, act='sigmoid')
        self.bn2 = BatchNorm(128)
        self.conv2_2 = Conv2D(128, 128, 3, padding=1, stride=1, act='sigmoid')
        self.bn2_2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 3, padding=1, stride=1, act='sigmoid')
        self.bn3 = BatchNorm(256)
        self.conv3_2 = Conv2D(256, 256, 3, padding=1, stride=1, act='sigmoid')
        self.bn3_2 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 3, padding=1, stride=1, act='sigmoid')
        self.bn4 = BatchNorm(512)
        self.conv4_2 = Conv2D(512, 512, 3, padding=1, stride=1, act='sigmoid')
        self.bn4_2 = BatchNorm(512)
        self.conv5 = Conv2D(512, 1024, 3, padding=1, stride=1, act='sigmoid')
        self.bn5 = BatchNorm(1024)
        self.conv5_2 = Conv2D(1024, 1024, 3, padding=1, stride=1, act='sigmoid')
        self.bn5_2 = BatchNorm(1024)
        self.conv6 = Conv2D(1024, 1024, 3, padding=1, stride=1, act='sigmoid')
        self.bn6 = BatchNorm(1024)
        self.conv6_2 = Conv2D(1024, 1024, 3, padding=1, stride=1, act='sigmoid')
        self.bn6_2 = BatchNorm(1024)

        self.fc1 = Linear(1024 * 7 * 7, 1024, act='sigmoid')
        self.fc2 = Linear(1024, num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.pool_down(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.pool_down(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)

        x = fluid.layers.reshape(x, [-1, 1024 * 7 * 7])
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class CNN_ConcatConv(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN_ConcatConv, self).__init__()

        self.conv1 = Conv2D(3, 64, 5, padding=2, stride=1, act='sigmoid')
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 5, padding=2, stride=1, act='sigmoid')
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 5, padding=2, stride=1, act='sigmoid')
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 5, padding=2, stride=1, act='sigmoid')
        self.bn4 = BatchNorm(512)
        self.block5 = ConcatConv(self.full_name(), (512, 384), (384, 256), (256, 256), 128)
        self.bn5 = BatchNorm(1024)
        self.conv6 = Conv2D(1024, 1024, 5, padding=2, stride=1, act='sigmoid')
        self.bn6 = BatchNorm(1024)

        self.fc1 = Linear(1024 * 7 * 7, 1024, act='sigmoid')
        self.fc2 = Linear(1024, num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.dropout = Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool_down(x)
        x = self.block5(x)
        x = self.bn5(x)
        x = self.pool_down(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = fluid.layers.reshape(x, [-1, 1024 * 7 * 7])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MyCNN(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(MyCNN, self).__init__()

        self.conv1 = Conv2D(3, 64, 5, padding=2, stride=1, act='leaky_relu')
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 5, padding=2, stride=1, act='leaky_relu')
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 5, padding=2, stride=1, act='leaky_relu')
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 5, padding=2, stride=1, act='leaky_relu')
        self.bn4 = BatchNorm(512)
        self.conv5 = Conv2D(512, 1024, 5, padding=2, stride=1, act='leaky_relu')
        self.bn5 = BatchNorm(1024)
        self.conv6 = Conv2D(1024, 1024, 5, padding=2, stride=1, act='leaky_relu')
        self.bn6 = BatchNorm(1024)

        self.fc1 = Linear(1024 * 7 * 7, 1024, act='leaky_relu')
        self.fc2 = Linear(1024, num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool_down(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool_down(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = fluid.layers.reshape(x, [-1, 1024 * 7 * 7])
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# 此模块将感受野为1×1、3×3、5×5的卷积层拼接在一起，使网络在这一层中就可以自动选择激活不同感受野的神经元
class ConcatConv(fluid.dygraph.Layer):
    def __init__(self, c1, c2, c3, c4, act_fun='sigmoid'):
        super(ConcatConv, self).__init__()

        self.p1_1 = Conv2D(c1[0], c1[1], 1, act=act_fun)
        self.p2_1 = Conv2D(c1[0], c2[0], 1, act=act_fun)
        self.p2_2 = Conv2D(c2[0], c2[1], 3, padding=1, act=act_fun)
        self.p3_1 = Conv2D(c1[0], c3[0], 1, act=act_fun)
        self.p3_2 = Conv2D(c3[0], c3[1], 5, padding=2, act=act_fun)
        self.p4_1 = Pool2D(pool_size=3, pool_stride=1, pool_padding=1, pool_type='max')
        self.p4_2 = Conv2D(c1[0], c4, 1, act=act_fun)

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))

        return fluid.layers.concat([p1, p2, p3, p4], axis=1)


class CNN_AllTricks(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(CNN_AllTricks, self).__init__()

        self.conv1 = Conv2D(3, 64, 3, padding=1, stride=1, act='leaky_relu')
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(64, 128, 3, padding=1, stride=1, act='leaky_relu')
        self.bn2 = BatchNorm(128)
        self.conv3 = Conv2D(128, 256, 3, padding=1, stride=1, act='leaky_relu')
        self.bn3 = BatchNorm(256)
        self.conv4 = Conv2D(256, 512, 3, padding=1, stride=1, act='leaky_relu')
        self.bn4 = BatchNorm(512)
        self.block5 = ConcatConv((512, 384), (384, 256), (256, 256), 128, act_fun='leaky_relu')
        self.bn5 = BatchNorm(1024)
        self.block6 = ConcatConv((1024, 384), (384, 256), (256, 256), 128, act_fun='leaky_relu')
        self.bn6 = BatchNorm(1024)

        self.pool_global = Pool2D(pool_stride=1, global_pooling=True, pool_type='avg')
        self.fc = Linear(input_dim=1024, output_dim=num_classes)

        self.pool_down = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.dropout = Dropout(p=0.5)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool_down(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool_down(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool_down(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool_down(x)
        x = self.block5(x)
        x = self.bn5(x)
        x = self.pool_down(x)
        x = self.block6(x)
        x = self.bn6(x)

        x = self.pool_global(x)
        x = fluid.layers.reshape(x, [-1, 1024])
        x = self.dropout(x)
        x = self.fc(x)

        return x


def train_MyCNN():
    epoch_num = 60
    batch_size = 32
    with fluid.dygraph.guard():
        model = CNN_AllTricks(num_classes=16)
        train(epoch_num, batch_size, model)


if __name__ == '__main__':
    train_MyCNN()
