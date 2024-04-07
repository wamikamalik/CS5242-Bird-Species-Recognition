#!/usr/bin/env python
# coding: utf-8

# ### Ensemble Classfiers

# In[63]:


import torch as torch
import random
from sklearn.metrics import classification_report
from random import randint


# #### Get the dataset

# In[64]:


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

# Load datasets
dataset_augmented = ImageFolder("../augmented_images", transform=transform)
dataset_images = ImageFolder("../CUB_200_2011/images", transform=transform)
dataset_augmented_more = ImageFolder("../augmented_images_more", transform=transform)
dataset_augmented_nocrop = ImageFolder("../augmented_images_nocrop", transform=transform)


# Combine datasets
all_classes = sorted(set(dataset_images.classes + dataset_augmented.classes + dataset_augmented_more.classes + dataset_augmented_nocrop.classes))
combined_dataset = ConcatDataset([dataset_images, dataset_augmented, dataset_augmented_more, dataset_augmented_nocrop])
combined_dataset.classes = all_classes

print("Loading test indices...")
test_indices = np.load('test_indices.npy')

# Create subset datasets
test_dataset = Subset(combined_dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Test set size: ", len(test_dataset))


# In[65]:


# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
dataset_augmented = ImageFolder("../augmented_images", transform=transform)
dataset_images = ImageFolder("../CUB_200_2011/images", transform=transform)
dataset_augmented_more = ImageFolder("../augmented_images_more", transform=transform)
dataset_augmented_nocrop = ImageFolder("../augmented_images_nocrop", transform=transform)


# Combine datasets
all_classes = sorted(set(dataset_images.classes + dataset_augmented.classes + dataset_augmented_more.classes + dataset_augmented_nocrop.classes))
combined_dataset = ConcatDataset([dataset_images, dataset_augmented, dataset_augmented_more, dataset_augmented_nocrop])
combined_dataset.classes = all_classes

# Create subset datasets
test_dataset_no_normalise = Subset(combined_dataset, test_indices)
test_loader_no_normalise = DataLoader(test_dataset_no_normalise, batch_size=32, shuffle=False)

print("No normalise test set size: ", len(test_dataset))


# #### Load the required models

# In[66]:


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
num_classes = len(combined_dataset.classes)
# model1 = torch.load(MODEL1_PATH, map_location=device)
# model1.eval()


# In[67]:


from torchvision.models import mobilenet_v2
import torch.nn.functional as F

class MobileNetModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.network = mobilenet_v2(pretrained=True)

        # No freezing
        # for param in self.network.parameters():
        #     param.requires_grad = True

        # Freeze all the parameters of the model
        for param in self.network.parameters():
            param.requires_grad = False

        # # Unfreeze the last five layers
        # for param in list(self.network.parameters())[-5:]:
        #     param.requires_grad = True

        # Unfreeze Batch Normalization
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

        # Unfreeze last layer
        for param in self.network.features[-1].parameters():
            param.requires_grad = True
                    
        # Replace the classifier with a custom one
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(0.5), 
		    nn.Linear(num_ftrs, num_classes)
            )
        # self.network.classifier[1] = nn.Linear(num_ftrs, num_classes)

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


# In[68]:


from torchvision.models import googlenet
import torch.nn.functional as F
        
class GoogLeNetModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.network = googlenet(pretrained=True)

        # Freeze all the parameters of the model
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze the last five layers
        for param in list(self.network.parameters())[-5:]:
            param.requires_grad = True

        # Replace the classifier with a custom one
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def unfreeze_all(self):
        for param in self.network.parameters():
            param.requires_grad = True
    
    def freeze_all_except_custom(self):
        # Freeze all the parameters of the model
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        list(self.network.parameters())[-1].requires_grad = True

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


# In[69]:


MODEL1_PATH = 'efficientnet_b0.pth'
MODEL2_PATH = 'googlenet_best.pth'
MODEL3_PATH = 'mobilenet_best.pth'


# In[71]:


# model1 = TheModelClass(*args, **kwargs)
# model1.load_state_dict(torch.load(MODEL1_PATH))
model1 = torch.load(MODEL1_PATH, map_location=device)
model1.eval()
model2 = torch.load(MODEL2_PATH, map_location=device)
model2.eval()
# model2 = MobileNetModel(num_classes)
# model2.load_state_dict(torch.load(MODEL2_PATH, map_location=device))
# model2 = model2.to(device)
# model2.eval()
model3 = MobileNetModel(num_classes)
model3.load_state_dict(torch.load(MODEL3_PATH, map_location=device))
model3 = model3.to(device)
model3.eval()


# In[72]:


model1_preds = [] # data predictions from model1
model2_preds = [] # data predictions from model2
model3_preds = [] # data predictions from model3
labels = []
with torch.no_grad():  # No need to track gradients
    for i in range(len(test_dataset)):
        inputs, l = test_dataset[i]
        inputs = inputs.to(device).unsqueeze(0)  # Convert inputs to tensor and send to device
        l = torch.tensor([l]).to(device)  # Convert label to tensor and send to device
        outputs = model1(inputs).cpu()
        model1_preds.extend(outputs)
        labels.extend(l.tolist())

with torch.no_grad():  # No need to track gradients
    for i in range(len(test_dataset_no_normalise)):
        inputs, l = test_dataset_no_normalise[i]
        inputs = inputs.to(device).unsqueeze(0)  # Convert inputs to tensor and send to device
        outputs = model2(inputs).cpu()
        model2_preds.extend(outputs)

with torch.no_grad():  # No need to track gradients
    for i in range(len(test_dataset_no_normalise)):
        inputs, l = test_dataset_no_normalise[i]
        inputs = inputs.to(device).unsqueeze(0)  # Convert inputs to tensor and send to device
        outputs = model3(inputs).cpu()
        model3_preds.extend(outputs)

# Expected outputs
num_of_classes = num_classes
expected_output_labels = labels # Expected output labels, assuming that all models are working with the same images
data_outputs = np.zeros((len(expected_output_labels), num_of_classes))
data_outputs[np.arange(len(expected_output_labels)), expected_output_labels] = 1
print("Expected output labels: ", expected_output_labels)
print("Data outputs length: ", data_outputs.shape)


# In[41]:


# Check if `model2_preds` contains arrays of different lengths
lengths = [len(pred) for pred in model1_preds]
if len(set(lengths)) > 1:
    print("model2_preds contains arrays of different lengths: ", lengths)
else:
    print("All arrays in model2_preds have the same length: ", lengths)

# Check if `model2_preds` contains arrays of different lengths
lengths = [len(pred) for pred in model2_preds]
if len(set(lengths)) > 1:
    print("model2_preds contains arrays of different lengths: ", lengths)
else:
    print("All arrays in model2_preds have the same length: ", lengths)

# Check if `model3_preds` contains arrays of different lengths
lengths = [len(pred) for pred in model3_preds]
if len(set(lengths)) > 1:
    print("model3_preds contains arrays of different lengths: ", lengths)
else:
    print("All arrays in model3_preds have the same length: ", lengths)

print("Model 1 prediction length: ", len(model1_preds))
print("Model 2 prediction length: ", len(model2_preds))
print("Model 3 prediction length: ", len(model3_preds))

from sklearn.metrics import accuracy_score

def test_model(preds, actual):
    '''
    Tests a model by calculating and printing its accuracy.
    '''
    # Convert probabilities to class labels
    preds_class = []
    for pred in preds:
        preds_class.append(np.argmax(pred))

    # Calculate the accuracy of the model
    accuracy = accuracy_score(actual, preds_class)

    # Print the accuracy of the model
    print("Accuracy: {:.2f}%".format(accuracy * 100))

print("Calculating accuracy for efficient net...")
test_model(model1_preds, expected_output_labels)
print("Calculating accuracy for google net...")
test_model(model2_preds, expected_output_labels)
print("Calculating accuracy for mobile net...")
test_model(model3_preds, expected_output_labels)


# #### Helper Functions

# In[53]:


import torch.nn.functional as F
def weighted_probability(num_of_classfiers, num_of_classes, networks_outputs, curr_weight_combi):
    '''
    Given an array contain the predictions from each classifier and the weights to be assigned to each classifier, 
    this function computes the final weighted probability.
    '''
    result = [0 for i in range(0, num_of_classes)]
    sum_of_weights = 0

    for i in range(0, num_of_classfiers):
        curr_network_output = networks_outputs[i]
        curr_weight = curr_weight_combi[i]
        sum_of_weights += curr_weight

        for j in range(0, num_of_classes):
            result[j] += curr_network_output[j] * curr_weight
    
    for k in range(0, num_of_classes):
        result[k] = result[k] / sum_of_weights

    return result


def fitness(y_pred, y_true): 
    '''
    Calculates the negative log loss.
    '''
    y_pred_tensor = torch.from_numpy(y_pred).float()
    y_true_tensor = torch.from_numpy(y_true).float()
    return F.cross_entropy(y_pred_tensor, y_true_tensor)


def mutate(weight_combi):
    '''
    Randomly changes a given float number (up to 2%). 
    '''
    for i in range(0, len(weight_combi)):
        weight_combi[i] = weight_combi[i] * random.uniform(0.99, 1.01)
    
    return weight_combi


def cross_over(num_of_classifiers, parent_1, parent_2):
    '''
    Given 2 different possible weight combination, this function produces a final weight combination by randomly extracting 
    weight elements from either parent combinations.
    '''
    cut = random.randint(0, num_of_classifiers - 1)
    new_weight_combi = parent_1[:cut] 
    new_weight_combi.extend(parent_2[cut:])

    return new_weight_combi


def generate_possible_weight_combis(num_of_classifiers, num_of_combis, weight_limit):
    '''
    Produces combinations of weights that can be assigned to each of the classifiers. 
    '''
    print("Generating possible weight combinations...")
    possible_weight_combis = []

    while (num_of_combis > 0):
        curr_weight_combi = []

        while (len(curr_weight_combi) < num_of_classifiers):
            curr_weight = random.uniform(0, weight_limit)
            curr_weight_combi.append(curr_weight)
        
        possible_weight_combis.append(curr_weight_combi)
        num_of_combis -= 1
    
    return possible_weight_combis


# #### Genetic Algorithm to find the optimal weights for each classifier
# 
# ![image.png](attachment:image.png)
# 
# Source: https://iopscience.iop.org/article/10.1088/2632-2153/ad10cf#mlstad10cfs2

# In[54]:


# Defining essential variables
num_of_classifiers = 3
num_of_required_weight_combis = 20
num_of_classes = num_classes
weight_limit = 100
possible_weight_combis = generate_possible_weight_combis(num_of_classifiers, num_of_required_weight_combis, weight_limit)
max_num_of_iters = 10

while (max_num_of_iters > 0):
    # Step 1: Randomly chossing 50% of the dataset to calculate the fitness scores for
    chosen_y_true = []
    chosen_y_model1_pred = []
    chosen_y_model2_pred = []
    chosen_y_model3_pred = []

    required_num_of_samples = len(data_outputs) // 2 # Rounding down

    random_indices = []
    while required_num_of_samples > 0:
        curr_index = randint(0, len(data_outputs) - 1)

        if (curr_index not in random_indices):
            chosen_y_true.append(data_outputs[curr_index])
            chosen_y_model1_pred.append(model1_preds[curr_index])
            chosen_y_model2_pred.append(model2_preds[curr_index])
            chosen_y_model3_pred.append(model3_preds[curr_index])

            random_indices.append(curr_index)
            required_num_of_samples -= 1

    # Step 2: Calculate the average fitness scores for each of the possible weight combinations
    fitness_and_weights = []
    print("Calculating fitness scores for each weight combination...")

    for weights in possible_weight_combis:
        accumulated_fitness_score = 0
        num_of_samples = len(chosen_y_true)

        for i in range(0, num_of_samples):
            network_outputs = [chosen_y_model1_pred[i], chosen_y_model2_pred[i], chosen_y_model3_pred[i]]
            y_pred = np.array(weighted_probability(num_of_classifiers, num_of_classes, network_outputs, weights))
            y_true = np.array(chosen_y_true[i])
            fitness_score = fitness(y_pred, y_true)
            accumulated_fitness_score += fitness_score
        
        avg_fitness_score = accumulated_fitness_score / num_of_samples
        fitness_and_weights.append((avg_fitness_score, weights))

    # Step 3: Rank the weight combis from best to worse
    print("Ranking the weight combinations from best to worst...")
    fitness_and_weights.sort() # The combis with the lowest log loss is at the start

    # Step 4: Selecting parents
    parents = []
    curr_index = 0

    print("Selecting parents...")

    # Selecting top 20% of the weight combis
    top_20_percent = int(len(fitness_and_weights) // 5) # Rounding down
    while (top_20_percent > 0):
        parents.append(fitness_and_weights[curr_index][1])
        top_20_percent -= 1
        curr_index += 1

    # Randomly choosing another 10% of the weight combinations
    another_10_percent = int(len(fitness_and_weights) // 10)  # Rounding down
    while(another_10_percent > 0):
        random_score_and_parent = random.choice(fitness_and_weights[curr_index:])
        parents.append(random_score_and_parent[1])
        fitness_and_weights.remove(random_score_and_parent)

        another_10_percent -= 1

    # Step 5: Randomly mutate 5% of the selected parents
    num_of_parents_to_mutate = max(1, int(len(parents) // 20))  # Rounding down
    index_of_parents_to_mutate = [random.randint(0, len(parents) - 1) for i in range(0, num_of_parents_to_mutate)]

    for index in index_of_parents_to_mutate:
        parents[index] = mutate(parents[index])

    # Step 6: Randomly cross over parents to produce new set of weight combinations
    print("Crossing over parents to produce new set of weight combinations...")
    new_weight_combis = []
    index_of_crossed_parents = []
    num_of_curr_weights = 0

    while (num_of_curr_weights < num_of_required_weight_combis):
        chosen_parents = (random.randint(0, len(parents) - 1), random.randint(0, len(parents) - 1))
        parent_1 = parents[chosen_parents[0]]
        parent_2 = parents[chosen_parents[1]]

        if (parent_1 != parent_2 and chosen_parents not in index_of_crossed_parents):
            new_weight_combi = cross_over(num_of_classifiers, parent_1, parent_2)
            new_weight_combis.append(new_weight_combi)
            num_of_curr_weights += 1

    possible_weight_combis = new_weight_combis
    print(possible_weight_combis) # For testing

    max_num_of_iters -= 1

# Step 7: Select the best weights combination
print("Selecting the best weight combination...")
final_fitness_and_weights = []

for weights in possible_weight_combis:
    accumulated_fitness_score = 0
    num_of_samples = 0

    for i in range(0, len(chosen_y_true)):
            network_outputs = [chosen_y_model1_pred[i], chosen_y_model2_pred[i], chosen_y_model3_pred[i]]
            y_pred = np.array(weighted_probability(num_of_classifiers, num_of_classes, network_outputs, weights))
            y_true = np.array(chosen_y_true[i])
            fitness_score = fitness(y_pred, y_true)
            accumulated_fitness_score += fitness_score

            num_of_samples += 1
    
    avg_fitness_score = accumulated_fitness_score / num_of_samples
    final_fitness_and_weights.append((avg_fitness_score, weights))

final_fitness_and_weights.sort() # The combis with the lowest log loss is at the start
best_weights = final_fitness_and_weights[0][1]
print("The best weight combination is: " + str(best_weights))
print("The fitness score of this combination is: " + str(final_fitness_and_weights[0][0]))


# #### Computing the performance of the ensemble model

# In[ ]:
from sklearn.metrics import accuracy_score

# Get final prediction of data based on the best weight combination
final_predictions = [weighted_probability(num_of_classifiers, num_of_classes, [model1_preds[i], model2_preds[i], model3_preds[i]], best_weights) for i in range(0, len(model1_preds))]
final_predictions = np.array(final_predictions)

# Get the final classification report
final_predictions = np.argmax(final_predictions, axis=1)
expected_output_labels = np.argmax(data_outputs, axis=1)
print("Accuracy Score: ", accuracy_score(expected_output_labels, final_predictions))
print(classification_report(expected_output_labels, final_predictions))

from sklearn.metrics import confusion_matrix

print("Saving confusion matrix...")
# Compute the confusion matrix
cm = confusion_matrix(expected_output_labels, final_predictions)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap of the confusion matrix
plt.figure(figsize=(120, 120))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Save the heatmap
plt.savefig("confusion_matrix_heatmap.png")

