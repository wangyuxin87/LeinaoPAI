import os
import shutil

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import transforms
from torchvision.datasets.mnist import read_image_file, read_label_file


# to use the data on our platform, we need to define a custom dataset
# this is a simple customization based on the original pytorch MNIST dataset
class MyMNIST(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            train_set = self.process()
            self.train_data, self.train_labels = train_set
        else:
            test_set = self.process()
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

    def process(self):
        print('Processing...')
        if self.train:
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
        return F.log_softmax(x, dim=1)


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    batch_idx = 0
    for batch_idx, (image, target) in enumerate(train_loader):
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= batch_idx
    train_accuracy = 100. * correct / len(train_loader.dataset)
    print("Epoch {} - Training: Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(epoch, train_accuracy, train_loss))


def val_one_epoch(model, device, val_loader, epoch=-1):
    model.eval()
    val_loss = 0
    correct = 0
    batch_idx = 0
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(val_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            val_loss += F.nll_loss(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= batch_idx
    val_accuracy = 100. * correct / len(val_loader.dataset)
    if epoch != -1:
        print("Epoch {} - Validation: Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(epoch, val_accuracy, val_loss))
    return val_accuracy, val_loss


def save_checkpoint(filename, model, checkpoint_best, is_best=0):
    if not os.path.exists('/userhome/checkpoints'):
        os.makedirs('/userhome/checkpoints')
    torch.save(model.state_dict(), filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_best)


def load_checkpoint(filename, model, device, val_loader):
    try:
        print("Loading checkpoint '{}'".format(filename))
        model.load_state_dict(torch.load(filename))
        avg_accuracy, avg_loss = val_one_epoch(model, device, val_loader)
        print("Checkpoint validation Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(avg_accuracy, avg_loss))
        return avg_accuracy

    except OSError:
        print("No specified checkpoint exists. Skipping...")
        print("**First time to train**")
        return 0


def main():
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
    device = 'cuda' if use_cuda else 'cpu'

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(
        MyMNIST(mnistdata_dir, train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = data.DataLoader(
        MyMNIST(mnistdata_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_acc = load_checkpoint(checkpoint_last, model, device, val_loader)

    for epoch in range(0, epochs):
        train_one_epoch(model, device, train_loader, optimizer, epoch)

        with torch.no_grad():
            acc, _ = val_one_epoch(model, device, val_loader, epoch)

        is_best = acc > best_acc
        if is_best:
            best_acc = acc

        save_checkpoint(checkpoint_last, model,
                        checkpoint_best, is_best=is_best)


if __name__ == '__main__':
    main()
