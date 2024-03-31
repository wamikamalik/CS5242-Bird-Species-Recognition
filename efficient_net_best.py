#!/usr/bin/env python

# coding: utf-8

# ## Loading the Data

# In[ ]:


import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import tarfile
import os

# Load the metadata
metadata = pd.read_csv('meta_data.csv')

# Create a dictionary mapping from image file name to is_training_image
is_training_image = dict(zip(metadata.augmented_image_name, metadata.is_training_image))


directory = "../CUB_200_2011/images"
cropped_directory = '../cropped_images'

# Check if the directory exists
if os.path.isdir(directory):
    print("Directory exists skipping extraction")
else:
    print("Directory does not exist, extracting tgz file...")

    # Specify the tgz file name
    tgz_file = "../CUB_200_2011.tgz"

    # Specify the directory to extract to
    extract_dir = "../"

    # Open the tgz file in read mode
    with tarfile.open(tgz_file, 'r:gz') as tar:
        # Extract all the contents of the tgz file in the specified directory
        tar.extractall(extract_dir)

if(os.path.isdir(cropped_directory)):
    print("Cropped directory exists, skipping cropping")

else:
    os.mkdir(cropped_directory)

    from PIL import Image

    print("Cropping the images...")

    # Create a dictionary mapping from image file name to bounding box
    bounding_boxes = dict(zip(metadata.augmented_image_name, zip(metadata.bounding_x, metadata.bounding_y, metadata.bounding_x + metadata.bounding_width, metadata.bounding_y + metadata.bounding_height)))

    # Function to crop an image using a bounding box
    def crop_image(img_path, bbox):
        with Image.open(img_path) as img:
            cropped_img = img.crop(bbox)
        return cropped_img

    # Crop each image using the bounding box
    cropped_images = [(crop_image(os.path.join(directory, img), bbox), label, img.split('/')[1]) for img, bbox, label in zip(metadata.image_name, bounding_boxes.values(), metadata.class_name)]

    print("Saving the cropped images...")

    # Save the cropped images to disk
    for i, (img, label, name) in enumerate(cropped_images):
        img_dir = os.path.join(cropped_directory, str(label))
        os.makedirs(img_dir, exist_ok=True)
        img.save(os.path.join(img_dir, name))

print("Loading the dataset...")
# Load the dataset
dataset = ImageFolder(cropped_directory)

print("Splitting the dataset...")
# Split the dataset into training and testing sets
train_indices = [i for i, (img, label) in enumerate(dataset.imgs) if is_training_image[os.path.normpath(img).replace('\\', '/').replace("../", "./").replace("./cropped_images", "./augmented_images")]]
test_indices = [i for i, (img, label) in enumerate(dataset.imgs) if not is_training_image[os.path.normpath(img).replace('\\', '/').replace("../", "./").replace("./cropped_images", "./augmented_images")]]

# ## Split Training Dataset into train and validation

# In[ ]:


from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset, DataLoader
import numpy as np

# Define transformations for training and test sets, without Resize here, since cropping is first
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get the labels of the test data
test_labels = np.array(dataset.targets)[test_indices]

# Split the test indices into extra and remaining subsets
extra_indices, remaining_indices = train_test_split(
    test_indices, test_size=0.5, stratify=test_labels)

# Split the remaining indices into validation and test subsets
val_indices, test_indices = train_test_split(
    remaining_indices, test_size=0.5, stratify=np.array(dataset.targets)[remaining_indices])

# Create subset datasets
train_dataset = Subset(dataset, train_indices)
extra_dataset = Subset(dataset, extra_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Apply transform to the datasets
train_dataset.dataset.transform = train_transform
extra_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

# Create the data loader for training
train_loader = DataLoader(ConcatDataset([train_dataset, extra_dataset]), batch_size=32, shuffle=True)

# Create the data loaders for validation and test
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Training set size: ", len(train_dataset) + len(extra_dataset))
print("Validation set size: ", len(val_dataset))
print("Test set size: ", len(test_dataset))


# ## EfficientNetB0 Transfer Learning Model Creation

# In[ ]:


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

        if epoch > 10:
            # Freeze all the parameters of the model
            for param in self.network.parameters():
                param.requires_grad = False

            for param in list(self.network.parameters())[-2:]:
                param.requires_grad = True

class EfficientNetModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.network = efficientnet_b0(pretrained=True)

        # Unfreeze all the parameters of the model
        for param in self.network.parameters():
            param.requires_grad = True

        # Unfreeze the last layer
        #for param in list(self.network.parameters())[-2:]:
        #    param.requires_grad = True

        # Add custom layers
        self.network.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, num_classes)
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


# ## Training the Model

# In[ ]:


import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(dataset.classes)
model = EfficientNetModel(num_classes).to(device)

def fit(epochs, lr, train_loader, val_loader, optimizer, weight_decay=0):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    optimizer = optimizer(model.parameters(), lr)
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
        print(f'Epoch {epoch+1}/{epochs}, train loss: {np.mean(train_losses):.4f}, val loss: {np.mean(val_losses):.4f}, train acc: {np.mean(train_accs):.4f}, val acc: {np.mean(val_accs):.4f}')
    return history


# In[ ]:


num_epochs = 20
opt_func = torch.optim.Adam
lr = 0.001
weight_decay = 1e-5
history = fit(num_epochs, lr, train_loader, val_loader, opt_func)


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

