import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from SnoutNet import SnoutNet
from SnoutNetData import SnoutNetData
import os
import numpy as np
from AlexNet import AlexNet
from VGG16 import VGG16
import argparse


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", type=str, default="none", help="Data model: alexnet | vgg16 | ensemble")
    argParser.add_argument("-w", type=str, required=True, default="none", help="weights file name")
    args = argParser.parse_args()

    models = []
    model = None

    weights_path = args.w

    if args.m == 'ensemble':
        models = [SnoutNet(), AlexNet(), VGG16()]
        models[0].load_state_dict(torch.load('snoutnet_weights_augmented.pth', map_location=device, weights_only=False))
        models[1].load_state_dict(torch.load('alexnet_weights_augmented.pth', map_location=device, weights_only=False))
        models[2].load_state_dict(torch.load('vgg16_weights_augmented.pth', map_location=device, weights_only=False))
        models[0].to(device)
        models[1].to(device)
        models[2].to(device)
    elif args.m == 'alexnet':
        model = AlexNet()
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
    elif args.m == 'vgg16':
        model = VGG16()
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
    else:
        model = SnoutNet()
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()

    # Transform default images to 227x227 and normalize
    transform = T.Compose([
        T.Resize((227, 227)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    lab_directory = os.path.dirname(__file__)
    data_directory = os.path.join(lab_directory, 'oxford-iiit-pet-noses')

    # Load dataset (use train or test)
    train_set = SnoutNetData(
        path=os.path.join(data_directory, 'images-original/images'),
        labels_file=os.path.join(data_directory, 'train_noses.txt'),
        transform=transform
    )

    test_set = SnoutNetData(
        path=os.path.join(data_directory, 'images-original/images'),
        labels_file=os.path.join(data_directory, 'test_noses.txt'),
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

    # Store distances to compare prediction vs actual
    distances = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if(args.m == 'ensemble'):
                ensemble_pred = []
                for m in models:
                    ensemble_pred.append(m(imgs))
                preds = torch.stack(ensemble_pred).mean(dim=0)
            else:
                preds = model(imgs)
            # scale to 227x227
            preds_pix = preds * 227
            labels_pix = labels * 227
            # Euclidean distance
            diff = preds_pix - labels_pix
            dist = torch.sqrt(torch.sum(diff ** 2, dim=1))
            distances.extend(dist.cpu().numpy())

    # Compute statistics
    distances = np.array(distances)
    min_dist = np.min(distances)
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    std_dist = np.std(distances)

    print("\nLocalization Accuracy (in pixels) for entire dataset:")
    print(f"  Min distance:  {min_dist:.3f}")
    print(f"  Mean distance: {mean_dist:.3f}")
    print(f"  Max distance:  {max_dist:.3f}")
    print(f"  Std deviation: {std_dist:.3f}")

    sorted_distances = np.argsort(distances)

    #Get the lowest 4 distances and their statistics
    best_indices = sorted_distances[:4]
    best_distances = distances[best_indices]
    min_dist = np.min(best_distances)
    mean_dist = np.mean(best_distances)
    max_dist = np.max(best_distances)
    std_dist = np.std(best_distances)

    print("\nLocalization Accuracy (in pixels) for 4 Best Worst Distances:")
    print(f"  Min distance:  {min_dist:.3f}")
    print(f"  Mean distance: {mean_dist:.3f}")
    print(f"  Max distance:  {max_dist:.3f}")
    print(f"  Std deviation: {std_dist:.3f}")

    #Get the highest 4 distances and print their statistics
    worst_indices = sorted_distances[-4:]
    worst_distances = distances[worst_indices]
    min_dist = np.min(worst_distances)
    mean_dist = np.mean(worst_distances)
    max_dist = np.max(worst_distances)
    std_dist = np.std(worst_distances)

    print("\nLocalization Accuracy (in pixels) for 4 Worst Distances:")
    print(f"  Min distance:  {min_dist:.3f}")
    print(f"  Mean distance: {mean_dist:.3f}")
    print(f"  Max distance:  {max_dist:.3f}")
    print(f"  Std deviation: {std_dist:.3f}")

    # Show the requested image
    idx = 0
    while True:
        idx = input("Enter index (or -1 to exit) to see an image > ")
        idx = int(idx)
        if idx < 0 or idx >= len(test_set):
            break

        img, label = test_set[idx]
        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            if (args.m == 'ensemble'):
                ensemble_pred = []
                for m in models:
                    ensemble_pred.append(m(img_input))
                pred = torch.stack(ensemble_pred).mean(dim=0)
            else:
                pred = model(img_input)

        pred = pred.squeeze(0).cpu().numpy()
        label = label.numpy()

        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_np = img_np.clip(0, 1)

        pred_pixel = pred.copy()
        pred_pixel[0] *= 227
        pred_pixel[1] *= 227

        label_pixel = torch.tensor(label).clone()
        label_pixel[0] *= 227
        label_pixel[1] *= 227

        plt.imshow(img_np)
        plt.scatter(label_pixel[0], label_pixel[1], c='green', s=50, label='Ground Truth')
        plt.scatter(pred_pixel[0], pred_pixel[1], c='red', s=50, label='Prediction')
        plt.title(f"Sample {idx}")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
