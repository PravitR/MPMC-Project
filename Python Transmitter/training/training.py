import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from datetime import datetime
from models import FCMNIST
import time
import random
import argparse
import yaml
import os

# Helper to generate a unique name for the saved model file based on settings
def create_run_name(hyperparameters):
    runname = hyperparameters["runtag"] + '_lr' + str(hyperparameters["learning_rate"]) + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["WScale"] + "_" +hyperparameters["QuantType"] + "_" + hyperparameters["NormType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_bs" + str(hyperparameters["batch_size"]) + "_epochs" + str(hyperparameters["num_epochs"])
    return runname

def train_model(model, device, hyperparameters, train_data, test_data):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]

    # Create DataLoaders
    if hyperparameters["augmentation"]: 
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        # Load entire dataset into GPU if no augmentation
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
        entire_dataset = next(iter(train_loader))
        all_train_images, all_train_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

    # Test dataset - total at once
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    entire_test_set = next(iter(test_loader))
    all_test_images, all_test_labels = entire_test_set[0].to(device), entire_test_set[1].to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)    

    # Training Loop 
    for epoch in range(num_epochs):
        correct = 0
        train_loss = []
        start_time = time.time()

        if hyperparameters["augmentation"]:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
        else:
            # Manual shuffling for non-augmented GPU training
            indices = list(range(len(all_train_images)))
            random.shuffle(indices)
            for i in range(len(indices) // batch_size):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                images = torch.stack([all_train_images[i] for i in batch_indices])
                labels = torch.stack([all_train_labels[i] for i in batch_indices])
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

        scheduler.step()

        # --- Testing Loop (Check accuracy after every epoch) ---
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(all_test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += all_test_labels.size(0)
            correct += (predicted == all_test_labels).sum().item()

        epoch_time = time.time() - start_time
        testaccuracy = correct / total * 100
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {np.mean(train_loss):.4f} | Test Acc: {testaccuracy:.2f}% | Time: {epoch_time:.2f}s')

if __name__ == '__main__':
    # Load parameters
    paramname = 'trainingparameters.yaml'
    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_res = hyperparameters.get("image_resolution", 16)

    # Basic transforms - without augmentation
    transform = transforms.Compose([
        transforms.Resize((img_res, img_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transform)

    if hyperparameters["augmentation"]:
        augmented_transform = transforms.Compose([
            transforms.RandomRotation(degrees=hyperparameters["rotation1"]),  
            transforms.RandomAffine(degrees=hyperparameters["rotation2"], translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((img_res, img_res)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        augmented_train_data = datasets.MNIST(root='data', train=True, transform=augmented_transform)
        train_data = ConcatDataset([train_data, augmented_train_data])

    # Initialize Model
    model = FCMNIST(
        network_width1=hyperparameters["network_width1"], 
        network_width2=hyperparameters["network_width2"], 
        network_width3=hyperparameters["network_width3"], 
        image_res=img_res,
        QuantType=hyperparameters["QuantType"], 
        NormType=hyperparameters["NormType"],
        WScale=hyperparameters["WScale"],
        quantscale=hyperparameters["quantscale"]
    ).to(device)

    print('Starting training...')
    train_model(model, device, hyperparameters, train_data, test_data)

    # Save Model
    if not os.path.exists('modeldata'):
        os.makedirs('modeldata')
        
    runname = create_run_name(hyperparameters)
    save_path = f'modeldata/{runname}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to: {save_path}')