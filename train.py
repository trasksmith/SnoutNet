import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import DataLoader
from SnoutNet import SnoutNet
from AlexNet import AlexNet
from VGG16 import VGG16
from SnoutNetData import SnoutNetData
import argparse
import torch
import os

#Default parameters
n_epochs = 50
batch_size = 32
learning_rate = 1e-4

def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, n_epochs, save_file, plot_file):
    print('Starting training...')

    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):

        epoch_train_loss = 0.0

        #Train the model
        model.train()
        for imgs, coords in train_loader:
            imgs = imgs.to(device)
            coords = coords.to(device)

            outputs = model(imgs)

            loss = loss_fn(outputs, coords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses += [epoch_train_loss/len(train_loader)]

        #Validate the model
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for imgs, coords in val_loader:
                imgs, coords = imgs.to(device), coords.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, coords)
                epoch_val_loss += loss.item()

        val_losses += [epoch_val_loss/len(val_loader)]
        scheduler.step(epoch_val_loss)

        print(f'{datetime.datetime.now()} | Epoch {epoch}/{n_epochs} | 'f'Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}')

    # Save weights and loss plot
    if save_file != None:
        torch.save(model.state_dict(), save_file)

    if plot_file != None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.savefig(plot_file)
        plt.close()

    print('Training complete! Model saved as', save_file)



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument("-a", type=str, default="none", help="Data augmentation type: none | all | color | geo")
    argParser.add_argument("-m", type=str, default="none", help="Data model: alexnet | vgg16 | ensemble")
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    args = argParser.parse_args()

    #Get weights and plot file from user input
    if args.s != None:
        save_file = args.s
    else:
        save_file = 'snoutnet_weights.pth'

    if args.p != None:
        plot_file = args.p
    else:
        plot_file = 'snoutnet_loss.png'

    #Choose augments depening on users input
    '''if args.a == 'all':
        #Add both geometric transforms and color transforms
        transform = T.Compose([
            T.Resize((227, 227)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop((216, 216)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.1),
            T.Resize((227, 227)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    elif args.a == 'color':
        #Add color transforms
        transform = T.Compose([
            T.Resize((227, 227)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    elif args.a == 'geo':
        #Add geometric transforms
        transform = T.Compose([
            T.Resize((227, 227)),
            T.RandomCrop((216, 216)),
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((227, 227)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    else:
        #Add no transforms other than normalizing
        transform = T.Compose([
            T.Resize((227, 227)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])'''
    
    if args.a == 'all':
        transform = T.Compose([
            T.Resize((227, 227)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        geo_transform = True
    elif args.a == 'geo':
        transform = T.Compose([
            T.Resize((227, 227)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        geo_transform = True
    elif args.a == 'color':
        transform = T.Compose([
            T.Resize((227, 227)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        geo_transform = False
    else:
        transform = T.Compose([
            T.Resize((227, 227)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        geo_transform = False

    #Chooe model depending on user input
    if args.m == 'alexnet':
        model = AlexNet().to(device)
    elif args.m == 'vgg16':
        model = VGG16().to(device)
    else:
        model = SnoutNet().to(device)

    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)
    print('\t\taugments  = ', args.a)
    print('\t\tmodel     = ', args.m)

    #Declare default validation transform since this isn't being trained it doesn't need to be changed
    val_transform = T.Compose([
            T.Resize((227, 227)),  # resize all images to 227x227
            T.ToTensor(),  # convert PIL image to PyTorch tensor (C×H×W)
            T.Normalize(mean=[0.485, 0.456, 0.406],  # normalize like ImageNet
                        std=[0.229, 0.224, 0.225])
    ])

    lab_directory = os.path.dirname(__file__)
    data_directory = os.path.join(lab_directory, 'oxford-iiit-pet-noses')

    #Use the train dataset for training and test dataset for validation
    train_dataset = SnoutNetData(
        path=os.path.join(data_directory, 'images-original/images'),
        labels_file=os.path.join(data_directory, 'train_noses.txt'),
        transform=transform,
        geo_transform=geo_transform
    )

    val_dataset = SnoutNetData(
        path=os.path.join(data_directory, 'images-original/images'),
        labels_file=os.path.join(data_directory, 'test_noses.txt'),
        transform=transform,
        geo_transform=False
    )

    #Declare all the required settings for training the model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, n_epochs, save_file, plot_file)

if __name__ == '__main__':
    main()