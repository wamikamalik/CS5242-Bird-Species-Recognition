import os
import sys
print(sys.path)

import numpy as np
import torch
from efficient_net.model import EfficientNetModel, GoogleNetModel, MobileNetModel

## for datasets and loaders
from data.data import get_annotations_file, get_dataset_dict, get_dataloader_dict
from train_test.test import plot_auroc, test_model, evaluate_model
from data.data import IDDataset

# metrics 
from torcheval.metrics import BinaryAUROC
from utils.utils import AP

#### argparser

## argparse 0. argparser - epochs, modes, set size, beta, m fix 0.4. 

import argparse

parser = argparse.ArgumentParser(description="Efficient Net")
parser.add_argument("--model", type=str, default="efficient", choices=["efficient", "google", "mobile"],
                    help = "efficient") 
parser.add_argument("--epoch", type = int, default=30,
                    help="epochs")
parser.add_argument("--temperature", type = int, default=1000,
                    help="temp")
parser.add_argument("--margin", type = float, default=0.4,
                    help="margin m")
parser.add_argument("--data", type=int,  default=4,
                    help = "data volume")
parser.add_argument("--mode", type=str, default="OOD",
                    help = "ood or normal")
parser.add_argument("--idbs", type=int, default=64,
                    help = "id batch size")
parser.add_argument("--oodbs", type=int, default=16,
                    help = "ood batch size")
parser.add_argument("--dir", type=str, default="nscc", choices=["local", "nscc", "custom"],
                    help = "img dir")
parser.add_argument("--beta", type=float, default=0.5,
                    help = "beta hp") 
parser.add_argument("--eps", type=float, default=10e-5,
                    help = "beta hp") 

args = parser.parse_args()

print(f"beta chosen: {args.beta}")


def fit(model, epochs, lr, train_loader, val_loader, oe_loader, optimizer, mode = "OOD"):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'auroc':[],
               'auprin': [], 'auprout': [], 'tpr95':[]}
    optimizer = optimizer(model.parameters(), lr)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        train_accs = []
        print("Starting epoch ", epoch+1, " of ", epochs)
        counter = 1
        
        # iterating thru batches

        for batch, batch_ood in zip(train_loader, oe_loader): #### consider tqdm....
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
                        
            inputs_ood, labels_ood = batch_ood
            inputs_ood = inputs_ood.to(device)
            labels_ood = labels_ood.to(device)
            
            # Compute predictions and losses
            outputs = model(inputs)
            loss, id_loss = model.training_step(batch, batch_ood, device, mode="OOD", margin = args.margin)
            train_losses.append(loss.item())
            train_losses.append(id_loss.item())
            # train_losses.append(mixed_loss.item())

            # Compute train accuracy
            _, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == labels.data) / len(labels)
            train_accs.append(acc.item())
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter+=1
        # Record training loss and accuracy
        history['train_loss'].append(torch.mean(torch.Tensor(train_losses)))
        history['train_acc'].append(torch.mean(torch.Tensor(train_accs)))

        # Validation phase
        model.eval()
        val_losses = []
        val_accs = []
        metric_dict = {'auroc': BinaryAUROC(),
                       'auprin': AP(),
                       'auprout': AP()}
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Compute predictions and losses
                loss_dict = model.validation_step(batch, device, args.temperature, metric_dict)
                val_losses.append(loss_dict['val_loss'].item())
                val_accs.append(loss_dict['val_acc'].item())
                metric_dict = loss_dict['metric_dict']
                # tpr95_ls.append(loss_dict['tpr95'].item())
        # Record validation loss and accuracy
        history['val_loss'].append(torch.mean(torch.Tensor(val_losses)))
        history['val_acc'].append(torch.mean(torch.Tensor(val_accs)))
        history['auroc'].append(metric_dict['auroc'].compute())
        history['auprin'].append(metric_dict['auprin'].compute(device))
        history['auprout'].append(metric_dict['auprout'].compute(device, mode = "out"))
        # history['tpr95'].append(np.mean(auprin_ls))

        print(f'Epoch {epoch+1}/{epochs}, train loss: {np.mean(train_losses):.4f}, val loss: {np.mean(val_losses):.4f}, train acc: {np.mean(train_accs):.4f}, val acc: {np.mean(val_accs):.4f}, auroc: {history['auroc'][-1]:.4f}, auprin: {history['auprin'][-1]:.4f}, auprout: {history['auprout'][-1]:.4f}')
        #  tpr95: {np.mean(tpr95_ls):.4f}
    return history

if __name__ == "__main__":
    ## pls convrt below to arg parse asap
    opt_func = torch.optim.Adam
    lr = 0.001

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### check if gpu available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ## loading model
    if args.model == "efficient":
        model = EfficientNetModel(160, args.beta).to(device)
    elif args.model == "google":
        model = GoogleNetModel(160, args.beta).to(device)
    elif args.model == "mobile":
        model = MobileNetModel(160, args.beta).to(device)
    # if args.model == "google":
    #     model = EfficientNetModel(160).to(device)
    # if args.model == "mobile":
    #     model = EfficientNetModel(160).to(device)

    if torch.cuda.device_count() > 1:
        ## applying data parallel - not suitable for now
        model = torch.nn.DataParallel(model)
        model = model.module
    ## loading in data
    df_chosen = get_annotations_file()
    dataset_dict = get_dataset_dict(df_chosen, args.data, data_dir_mode=args.dir)
    dataloader_dict = get_dataloader_dict(dataset_dict)
    print(dataloader_dict.keys())

    # setting names
    print(f"epochs ={args.epoch}, mode = {args.mode}")
    history_name = "history_" + args.model +args.mode + ".png"
    model_name = args.model + args.mode + ".pth"
    print(f"history name: {history_name}, model name: {model_name}")

    history = fit(model, args.epoch, lr, dataloader_dict['train'], dataloader_dict['final_val'],
                   dataloader_dict['oe'], opt_func, mode="normal")
    
    # converting history tensors to cpu
    for key in history:
        history[key] = [metric.cpu() for metric in history[key]]

    # Save the trained model
    torch.save(model, model_name)

    plot_auroc(history)
    test_model(model, dataloader_dict['test'], device) # specifically for ID data
    evaluate_model(model, dataloader_dict["final_val"], device, args.temperature)
    ### testing stage