import os
import shutil

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets.mnist import read_image_file, read_label_file

from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss


# to use the data on our platform, we need to define a custom dataset
# this is a simple customization based on the original pytorch MNIST dataset
class MyMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            train_set = self.process(self.train)
            self.train_data, self.train_labels = train_set
        else:
            test_set = self.process(self.train)
            self.test_data, self.test_labels = test_set

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def process(self, train):
        print('Processing...')
        if train:
            training_set = (
                read_image_file(os.path.join(
                    self.root, 'train-images-idx3-ubyte')),
                read_label_file(os.path.join(
                    self.root, 'train-labels-idx1-ubyte'))
            )
            print('Done!')
            return training_set
        else:
            test_set = (
                read_image_file(os.path.join(
                    self.root, 't10k-images-idx3-ubyte')),
                read_label_file(os.path.join(
                    self.root, 't10k-labels-idx1-ubyte'))
            )
            print('Done!')
            return test_set


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


if __name__ == "__main__":

    # configs
    lr = 0.01
    momentum = 0.5
    epochs = 10
    batch_size = 64
    mnistdata_dir = '/gdata/MNIST'
    checkpoint_dir = '/userhome/checkpoints'
    checkpoint_last = os.path.join(checkpoint_dir, "mnist_last.pth")
    checkpoint_best = os.path.join(checkpoint_dir, "mnist_best.pth")

    use_cuda = torch.cuda.is_available()

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MyMNIST(mnistdata_dir, train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        MyMNIST(mnistdata_dir, train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = Net()
    device = 'cuda' if use_cuda else 'cpu'
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(
        model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': CategoricalAccuracy(),
                 'nll': Loss(F.nll_loss)},
        device=device)

    @trainer.on(Events.STARTED)
    def load_checkpoint(engine):
        # you can load the best checkpoint to continue training
        filename = checkpoint_best
        # or load the last checkpoint
        filename = checkpoint_last
        try:
            print("Loading checkpoint '{}'".format(filename))
            model.load_state_dict(torch.load(filename))
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            print("Checkpoint loaded successfully from '{}' "
                  .format(filename))
            print("Checkpoint validation Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(avg_accuracy, avg_nll))
            engine.state.best_acc = avg_accuracy

        except OSError:
            print("No specified checkpoint exists. Skipping...")
            print("**First time to train**")
            engine.state.best_acc = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Epoch {} - Training: Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Epoch {} - Validation: Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        engine.state.acc = avg_accuracy

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(model.state_dict(), checkpoint_last)
        is_best = engine.state.acc > engine.state.best_acc
        if is_best:
            engine.state.best_acc = engine.state.acc
            shutil.copyfile(checkpoint_last, checkpoint_best)

    trainer.run(train_loader, max_epochs=epochs)
