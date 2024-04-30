import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.utils import auprIn, auprOut
import torch.nn.functional as F

# metrics 
from torcheval.metrics import BinaryAUROC
from utils.utils import AP
from sklearn import metrics

def plot_auroc(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(history['auroc'], label='auroc')
    ax1.plot(history['auprin'], label='auprin')
    ax1.plot(history['auprout'], label='auprout')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    plt.tight_layout()
    plt.savefig('history_auroc.png')

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # # defining top k accuracy from author
            # acc1, acc5 = accuracy_author(output, y, topk=(1, 5))
            # top1.update(acc1, x.size(0))
            # top5.update(acc5, x.size(0))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

def evaluate_model(model, test_dataloader, device, temperature):
    model.eval()
    val_losses = []
    val_accs = []

    with torch.no_grad():

        metric_dict = {'auroc': BinaryAUROC(),
                       'auprin': AP(),
                       'auprout': AP()}
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Compute predictions and losses
                loss_dict = model.validation_step(batch, device, temperature, metric_dict)
                val_losses.append(loss_dict['val_loss'].item())
                val_accs.append(loss_dict['val_acc'].item())
                metric_dict = loss_dict['metric_dict']

        #plotting auroc

        inputs = torch.cat(metric_dict['auroc'].inputs)
        print(inputs.shape)
        targets = torch.cat(metric_dict['auroc'].targets)
        fpr, tpr, thresholds = metrics.roc_curve(targets, inputs, pos_label=1)
        print(fpr)
        print(tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metric_dict['auroc'].compute():.2f}) (auprin = {metric_dict['auprin'].compute(device):.2f}) (auprout = {metric_dict['auprout'].compute(device, mode = "out"):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('AUROC.png')