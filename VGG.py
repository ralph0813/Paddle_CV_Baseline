import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Dropout, BatchNorm

from train import train


class VGG13(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(VGG13, self).__init__()

        self.drop_out = Dropout(p=0.5)
        self.pool = Pool2D(pool_size=3, pool_stride=2, pool_type='max')

        self.conv1 = Conv2D(num_channels=3, num_filters=64, filter_size=3, stride=1, padding=1, act='relu')

        self.conv2 = Conv2D(num_channels=64, num_filters=128, filter_size=3, stride=1, padding=1, act='relu')

        self.conv3 = Conv2D(num_channels=128, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        self.conv4 = Conv2D(num_channels=256, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')

        self.conv5 = Conv2D(num_channels=256, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        self.conv6 = Conv2D(num_channels=512, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')

        self.conv7 = Conv2D(num_channels=512, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        self.conv8 = Conv2D(num_channels=512, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')

        self.fc1 = Linear(512 * 6 * 6, 4096, act='relu')
        self.fc2 = Linear(4096, 4096, act='relu')
        self.fc3 = Linear(4096, num_classes, act='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool(x)

        x = fluid.layers.reshape(x, [-1, 512 * 6 * 6])
        x = self.fc1(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.drop_out(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    epoch_num = 60
    batch_size = 32
    with fluid.dygraph.guard():
        model = VGG13(num_classes=16)
        train(epoch_num, batch_size, model)
