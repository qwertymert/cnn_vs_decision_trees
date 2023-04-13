from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

import matplotlib.pyplot as plt
import numpy as np

from dataset_tools import load_images_from_folder, scale_dataset

seed = 0

torch.manual_seed(seed)
np.random.seed(seed)


# create custom dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def create_dataloader(x, y, batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CustomDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader


# create pretrained resnet50 classification model
class ResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        model = torchvision.models.resnet50()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# learning rate scheduler
def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


# create training loop in main
def main():
    image_size = 128
    epochs = 50
    batch_size = 64
    learning_rate = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # load data
    (x_train_dict, y_train_dict), (x_val_dict, y_val_dict), (x_test_dict, y_test_dict) = load_images_from_folder(
        "main_dataset", image_size)

    x_train, y_train, scaler = scale_dataset(x_train_dict, y_train_dict, "standard", True, seed)
    x_val, y_val, _ = scale_dataset(x_val_dict, y_val_dict, "standard", True, seed, scaler)
    x_test, y_test, _ = scale_dataset(x_test_dict, y_test_dict, "standard", False, seed, scaler)

    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_val, y_val = torch.from_numpy(x_val), torch.from_numpy(y_val)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

    # create dataloaders
    train_dataloader = create_dataloader(x_train, y_train, batch_size=batch_size)
    val_dataloader = create_dataloader(x_val, y_val, batch_size=batch_size)
    test_dataloader = create_dataloader(x_test, y_test, batch_size=batch_size)

    model = ResNet50()

    model = model.to(device)

    print(model)

    # create loss function
    loss_fn = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # train model
    train_losses, train_accs = [], []

    val_losses, val_accs = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        print("-" * 10)

        # train model
        model.train()

        optimizer = lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=7)

        train_loss = 0
        train_correct = 0
        for x, y in train_dataloader:
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.int64)

            # forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate loss and accuracy
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # calculate loss and accuracy
        train_loss /= len(train_dataloader.dataset)
        train_acc = train_correct / len(train_dataloader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validate model
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.int64)

                # forward pass
                pred = model(x)
                loss = loss_fn(pred, y)

                # calculate loss and accuracy
                val_loss += loss.item()
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # calculate loss and accuracy
        val_loss /= len(val_dataloader.dataset)
        val_acc = val_correct / len(val_dataloader.dataset)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        # print loss and accuracy
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        print()

    # test model
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.int64)

            # forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            # calculate loss and accuracy
            test_loss += loss.item()
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # calculate loss and accuracy
    test_loss /= len(test_dataloader.dataset)
    test_acc = test_correct / len(test_dataloader.dataset)

    # print loss and accuracy
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print()

    # save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # plot loss
    plt.style.use("ggplot")
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Loss curves of ResNet50 model")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "val"])
    plt.savefig("resnet50_loss_curve")
    plt.show()

    # plot accuracy
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title("Accuracy curves of ResNet50 model")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["train", "val"])
    plt.savefig("resnet50_acc_curve")
    plt.show()


if __name__ == "__main__":
    main()