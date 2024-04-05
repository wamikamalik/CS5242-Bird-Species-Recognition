#!/usr/bin/env python
# coding: utf-8

# ## Loading the Data

# In[1]:


import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import zipfile
import os
import csv
from torch.utils.data import ConcatDataset
import numpy as np
# Extra info for logging
extra = 'change to freeze all layers + dropout 0.5 + unfreeze batchnorm + cropped images + track both best val_loss and val_acc + add original image + test image not cropped + add augmented_more dataset + no crop augmented data'

# Load the metadata
metadata = pd.read_csv('meta_data.csv')
metadata['augmented_images_more'] = metadata['augmented_image_name'].str.replace('augmented_images','augmented_images_more')
metadata['augmented_image_nocrop_name'] = metadata['augmented_image_name'].str.replace('augmented_images','augmented_images_nocrop')

# Create a dictionary mapping from image file name to is_training_image

# Only add training images for augmented
temp = metadata[metadata.is_training_image == 1]

is_training_image = dict(zip(temp.augmented_image_name, temp.is_training_image))   # original augmentation
is_training_image.update(zip(temp.augmented_images_more, temp.is_training_image))  # more augmentation
is_training_image.update(zip(temp.augmented_image_nocrop_name, temp.is_training_image)) # augmented but no crop to bbox

# Add original data
# Test will be from original images (not cropped, only resize to 224,224)
is_training_image.update(zip(metadata.image_name, metadata.is_training_image))

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load or extract dataset
def load_or_extract(directory, zip_file):
    if os.path.isdir(directory):
        print(f"Directory {directory} exists, skipping zip extraction.")
    else:
        print(f"Directory {directory} does not exist, extracting zip file {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("../")
        print(f"Extracted {zip_file}.")

# Load or extract both datasets
load_or_extract("../augmented_images", "../augmented_images.zip")
load_or_extract("../CUB_200_2011/images", "../CUB_200_2011/images.zip")
load_or_extract("../augmented_images_more","../augmented_images_more.zip")
load_or_extract("../augmented_images_nocrop","../augmented_images_nocrop.zip")

# Load datasets
dataset_augmented = ImageFolder("../augmented_images", transform=transform)
dataset_images = ImageFolder("../CUB_200_2011/images", transform=transform)
dataset_augmented_more = ImageFolder("../augmented_images_more", transform=transform)
dataset_augmented_nocrop = ImageFolder("../augmented_images_nocrop", transform=transform)


# Combine datasets
all_classes = sorted(set(dataset_images.classes + dataset_augmented.classes + dataset_augmented_more.classes + dataset_augmented_nocrop.classes))
combined_dataset = ConcatDataset([dataset_images, dataset_augmented, dataset_augmented_more, dataset_augmented_nocrop])
combined_dataset.classes = all_classes
print(combined_dataset)


# ## Split Training Dataset into train and validation

# In[5]:


from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset, DataLoader
import numpy as np

def get_combined_indices(combined_dataset, is_training_image):
    train_indices, test_indices = [], []
    offset = 0  # Offset to account for different datasets in ConcatDataset
    for dataset in combined_dataset.datasets:
        for i, (img, _) in enumerate(dataset.imgs):
            # Construct relative path as used in is_training_image
            rel_path = "./" + os.path.normpath(img)
            if rel_path in is_training_image:  # Check if the image is in the dictionary
                if is_training_image[rel_path] == 1:
                    train_indices.append(i + offset)  # Image is for training
                else:
                    test_indices.append(i + offset)  # Image is for testing
        offset += len(dataset)  # Update offset for the next dataset
    return train_indices, test_indices

if not os.path.exists('train_indices.npy'):
    # Get combined train and test indices
    train_indices, test_indices = get_combined_indices(combined_dataset, is_training_image)
    test_indices, val_indices = train_test_split(test_indices, test_size=0.2, random_state=42)
    print(f"Number of training indices: {len(train_indices)}")
    print(f"Number of testing indices: {len(test_indices)}")
    print(f"Number of testing indices: {len(val_indices)}")

    # Save the indices to .npy files
    np.save('train_indices.npy', train_indices)
    np.save('test_indices.npy', test_indices)
    np.save('val_indices.npy', val_indices)
else:
    print("indices exist, loading them...")

# Load indices from .npy files (demonstration)
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')
test_indices = np.load('test_indices.npy')

# Create subset datasets
train_dataset = Subset(combined_dataset, train_indices)
val_dataset = Subset(combined_dataset, val_indices)
test_dataset = Subset(combined_dataset, test_indices)

# Create the data loader for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create the data loaders for validation and test
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Training set size: ", len(train_dataset))
print("Validation set size: ", len(val_dataset))
print("Test set size: ", len(test_dataset))


# ## EfficientNetB0 Transfer Learning Model Creation

# In[12]:


import torch
from torch import nn
from torchvision.models import efficientnet_b0
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))

class EfficientNetModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.network = efficientnet_b0(pretrained=True)

        # Add custom layers
        self.network.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )


    def forward(self, xb):
        return self.network(xb)

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def freeze(self):
        # Freeze all the parameters of the model
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze the last layers
        for param in list(self.network.parameters())[0:-4]:
            param.requires_grad = True

    def unfreeze(self):
        # Unfreeze all parameters of the model
        for param in self.network.parameters():
            param.requires_grad = True


# ## Training the Model

# In[14]:


import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(combined_dataset.classes)
model = EfficientNetModel(num_classes).to(device)
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

def fit(epochs, lr, train_loader, val_loader, optimizer, weight_decay=0):
    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        train_accs = []
        print("Starting epoch ", epoch+1, " of ", epochs)
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch = (inputs, labels)
            # Compute predictions and losses
            outputs = model(inputs)
            loss = model.training_step(batch)
            train_losses.append(loss.item())
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == labels.data) / len(labels)
            train_accs.append(acc.item())
            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
        # Record training loss and accuracy
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(np.mean(train_accs))

        # Validation phase
        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch = (inputs, labels)
                # Compute predictions and losses
                loss = model.validation_step(batch)['val_loss']
                val_losses.append(loss.item())
                # Compute accuracy
                acc = model.validation_step(batch)['val_acc']
                val_accs.append(acc.item())
        # Record validation loss and accuracy
        history['val_loss'].append(np.mean(val_losses))
        history['val_acc'].append(np.mean(val_accs))
        print(f'Epoch {epoch+1}/{epochs}, train loss: {np.mean(train_losses):.4f}, val loss: {np.mean(val_losses):.4f}, val loss: {np.mean(val_losses):.4f}, train acc: {np.mean(train_accs):.4f}, val acc: {np.mean(val_accs):.4f}')
    return history


# In[15]:


num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001
weight_decay = 1e-5
model.freeze()
history = fit(num_epochs, lr, train_loader, val_loader, opt_func)
#model.unfreeze()
#lr = 0.0001
#new_history = fit(num_epochs, lr, train_loader, val_loader, opt_func)
# Append new history to existing history
#for key in history.keys():
#    history[key].extend(new_history[key])


# In[ ]:


import matplotlib.pyplot as plt

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(history['train_loss'], label='train loss')
    ax1.plot(history['val_loss'], label='validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='train accuracy')
    ax2.plot(history['val_acc'], label='validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('efficientnet_train_val_acc_loss.png')

plot_history(history)


# ## Test the Model

# In[ ]:


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

test_model(model, test_loader, device)


# ## Save the Model

# In[ ]:


# Save the trained model
torch.save(model, 'efficientnet_b0.pth')

