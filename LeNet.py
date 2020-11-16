import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Dropout, BatchNorm

from train import train


class LeNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(num_channels=3, num_filters=6, filter_size=5, act='relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=6, num_filters=16, filter_size=5, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv3 = Conv2D(num_channels=16, num_filters=120, filter_size=4, act='relu')

        self.fc1 = Linear(input_dim=120 * 50 * 50, output_dim=84, act='relu')
        self.fc2 = Linear(input_dim=84, output_dim=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = fluid.layers.reshape(x, [-1, 120 * 50 * 50])  # 将二维的卷积层输出的特征图拉伸为同等大小的1维
        x = self.fc1(x)
        y = self.fc2(x)
        return y


if __name__ == '__main__':
    epoch_num = 60
    batch_size = 32
    with fluid.dygraph.guard():
        model = LeNet(num_classes=16)
        train(epoch_num, batch_size, model)
