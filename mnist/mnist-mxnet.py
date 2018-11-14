import os
import shutil
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag

def train_one_epoch(epoch, optimizer, train_data, criterion, ctx):
    train_data.reset()
    acc_metric = mx.metric.Accuracy()
    loss_sum = 0.0
    count = 0
    for batch in train_data:

        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx,
                                          batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx,
                                           batch_axis=0)
        outputs = []
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                loss = criterion(z, y)
                loss_sum += float(loss.sum().asnumpy())
                count += data[0].shape[0]
                loss.backward()
                outputs.append(z)
                acc_metric.update(label, outputs)

        optimizer.step(batch.data[0].shape[0])

    _, avg_acc = acc_metric.get()
    avg_loss = loss_sum / count
    acc_metric.reset()

    print('Epoch {} - Training: Avg accuracy: {:.2f} '
          'Avg loss: {:.2f}'.format(epoch, 100.0*avg_acc, avg_loss))


def val_one_epoch(epoch, val_data, criterion, ctx):
    val_data.reset()
    acc_metric = mx.metric.Accuracy()
    loss_sum = 0.0
    count = 0
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []

        for x, y in zip(data, label):
            z = net(x)
            loss = criterion(z, y)
            loss_sum += float(loss.sum().asnumpy())
            count += data[0].shape[0]
            outputs.append(z)
            acc_metric.update(label, outputs)

    _, avg_acc = acc_metric.get()
    avg_loss = loss_sum / count
    acc_metric.reset()

    if epoch >= 0:
        print('Epoch {} - Validation: Avg accuracy: {:.2f} '
              'Avg loss: {:.2f}'.format(epoch, 100.0*avg_acc, avg_loss))
    return avg_acc


def save_checkpoint(net, checkpoint_dir, is_best=0):

    net.save_params(os.path.join(checkpoint_dir, 'mnist_last.params'))

    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, 'mnist_last.params'),
                        os.path.join(checkpoint_dir, 'mnist_best.params'))


def load_checkpoint(net, checkpoint_dir, val_data, criterion, ctx):
    best_acc = 0
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    else:
        if os.path.exists(os.path.join(checkpoint_dir,'mnist_best.params')):
            net.load_params(os.path.join(checkpoint_dir,'mnist_best.params'))
            best_acc = val_one_epoch(-1, val_data, criterion,  ctx)
            print('The last accuracy: {:.2f}'.format(100.0 * best_acc))
    return best_acc


if __name__ == '__main__':

    lr = 0.01
    momentum = 0.5
    epoch = 10
    batch_size = 64
    mnistdata_dir = '/gdata/MNIST'
    checkpoint_dir = '/userhome/checkpoints'

    train_data = mx.io.MNISTIter(
        image=os.path.join(mnistdata_dir, "train-images-idx3-ubyte"),
        label=os.path.join(mnistdata_dir, "train-labels-idx1-ubyte"),
        batch_size=batch_size,
        data_shape=(28, 28),
        shuffle = True
        )


    val_data = mx.io.MNISTIter(
        image=os.path.join(mnistdata_dir, "t10k-images-idx3-ubyte"),
        label=os.path.join(mnistdata_dir, "t10k-labels-idx1-ubyte"),
        batch_size=batch_size,
        data_shape=(28, 28),
        shuffle=False
        )

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=10,kernel_size=5,use_bias=False)),
        net.add(nn.MaxPool2D()),
        net.add(nn.LeakyReLU(0))

        net.add(nn.Conv2D(channels=20, kernel_size=5, use_bias=False)),
        net.add(nn.Dropout(0.5)),
        net.add(nn.MaxPool2D()),
        net.add(nn.LeakyReLU(0)),

        net.add(nn.Flatten()),
        net.add(nn.Dense(50, activation='relu')),
        net.add(nn.Dropout(0.5)),
        net.add(nn.Dense(10))

    gpus = mx.test_utils.list_gpus()
    ctx = [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum':momentum})

    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    best_acc = load_checkpoint(net, checkpoint_dir, val_data, softmax_cross_entropy_loss, ctx)

    print('start train...')
    for i in range(epoch):

        train_one_epoch(i, trainer, train_data, softmax_cross_entropy_loss, ctx)

        acc = val_one_epoch(i, val_data, softmax_cross_entropy_loss, ctx)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint(net, checkpoint_dir, is_best)
