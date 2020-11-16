from visualdl import LogWriter
import re
import os
import shutil
import paddle.fluid as fluid
import numpy as np

from reader import tra_loader, val_loader


def train(epoch_num, batch_size, model, log_dir_name='log', lr_decay=False):
    print("Start training.")
    train_loader = tra_loader(batch_size=batch_size)
    valid_loader = val_loader(batch_size=batch_size)

    log_dir = log_dir_name + '/' + re.search(r'^.*_', model.full_name()).group()[:-1]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    train_loss_wrt = LogWriter(logdir=log_dir + '/train_loss')
    train_acc_wrt = LogWriter(logdir=log_dir + '/train_acc')
    val_loss_wrt = LogWriter(logdir=log_dir + '/val_loss')
    val_acc_wrt = LogWriter(logdir=log_dir + '/val_acc')

    if not lr_decay:
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
    else:
        opt = fluid.optimizer.Adam(learning_rate=fluid.dygraph.ExponentialDecay(
            learning_rate=0.002,
            decay_steps=1000,
            decay_rate=0.1,
            staircase=True), parameter_list=model.parameters())

    for epoch in range(epoch_num):
        avg_loss_acc = np.zeros([6])

        model.train()
        for batch_id, data in enumerate(train_loader()):
            xd, yd = data

            img = fluid.dygraph.to_variable(xd)
            label = fluid.dygraph.to_variable(yd)
            logits = model(img)
            pred = fluid.layers.softmax(logits)
            loss = fluid.layers.softmax_with_cross_entropy(logits, label)
            avg_loss = fluid.layers.mean(loss)
            acc = fluid.layers.accuracy(input=pred, label=label)
            avg_loss_acc[0] += avg_loss.numpy()[0]
            avg_loss_acc[1] += acc.numpy()[0]
            avg_loss_acc[2] += 1
            # print('epoch:', epoch, ', batch:', batch_id, ', train loss:', avg_loss.numpy(), ', train acc:', acc.numpy())

            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
        # break

        model.eval()
        for batch_id, data in enumerate(valid_loader()):
            xd, yd = data
            img = fluid.dygraph.to_variable(xd)
            label = fluid.dygraph.to_variable(yd)
            logits = model(img)
            pred = fluid.layers.softmax(logits)
            loss = fluid.layers.softmax_with_cross_entropy(logits, label)
            avg_loss = fluid.layers.mean(loss)
            acc = fluid.layers.accuracy(input=pred, label=label)
            avg_loss_acc[3] += avg_loss.numpy()[0]
            avg_loss_acc[4] += acc.numpy()[0]
            avg_loss_acc[5] += 1
            # print('validation loss:', avg_loss.numpy(), ', validation acc:', acc.numpy())

        avg_loss_acc[0:2] = avg_loss_acc[0:2] / avg_loss_acc[2]
        avg_loss_acc[3:5] = avg_loss_acc[3:5] / avg_loss_acc[5]
        avg_loss_acc = np.around(avg_loss_acc, decimals=4)

        print('epoch:', epoch, 'train loss:', avg_loss_acc[0], ', train acc:', avg_loss_acc[1], ', validation loss:',
              avg_loss_acc[3], ', validation acc:', avg_loss_acc[4])
        train_loss_wrt.add_scalar(tag='train_loss', step=epoch, value=avg_loss_acc[0])
        train_acc_wrt.add_scalar(tag='train_acc', step=epoch, value=avg_loss_acc[1])
        val_loss_wrt.add_scalar(tag='val_loss', step=epoch, value=avg_loss_acc[3])
        val_acc_wrt.add_scalar(tag='val_acc', step=epoch, value=avg_loss_acc[4])