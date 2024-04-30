import torch
import numpy as np
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision.models import googlenet
from torchvision.models import mobilenet_v2
from utils.utils import auprIn, auprOut
from torch.distributions.beta import Beta
import torch.nn.functional as F


## efficient net b4 ahs 19mil vs b05mil

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
    def __init__(self, num_classes, beta=0.5):
        super().__init__()
        self.network = efficientnet_b0(pretrained=True)
        self.num_classes = num_classes
        self.beta = beta

        # # Freeze all the parameters of the model
        # for param in self.network.parameters():
        #     param.requires_grad = False

        # # Unfreeze the last three layers
        # for param in list(self.network.parameters())[-3:]:
        #     param.requires_grad = True

        # Find the last linear layer in the classifier
        last_linear_layer = None
        for layer in reversed(self.network.classifier):
            if isinstance(layer, nn.Linear):
                last_linear_layer = layer
                break

        # Check if a linear layer was found
        if last_linear_layer is not None:
            num_ftrs = last_linear_layer.in_features
        else:
            raise ValueError("No linear layer found in classifier")

        # Add custom layers
        self.network.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

    def training_step(self, batch, batch_ood, device, mode = "OOD", margin = 0.4):

        """
        mode can be "normal" or "OOD"
        run inference, generate losses, output losses
        """

        # generate prediction first, normal image and labels to device
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        images_ood, labels_ood = batch_ood
        images_ood = images_ood.to(device)
        labels_ood = labels_ood.to(device)

        if mode == "normal":
          out = self(images)                  #[bs]
          loss = F.cross_entropy(out, labels) # Calculate loss

          return loss

        elif mode == "OOD":

          """
          requires running on normal inference and OE MIX
          """

          ##### ID LOSS #######
          out_id = self(images) # [b, num_class]
          id_loss = F.cross_entropy(out_id, labels)

          ### overall loss ######
          # ref: https://github.com/YU1ut/Ensemble-of-Leave-out-Classifiers/blob/master/eloc_solver.py#L63

          # somehow F.log_softmax(out_id, dim=1) * F.softmax(out_id, dim=1) this is entropy

          E_id = -torch.mean(torch.sum(F.log_softmax(out_id, dim=1) * F.softmax(out_id, dim=1), dim=1))

          output_ood = self(images_ood) # [b, num_class]

          E_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))

          loss = F.cross_entropy(out_id, labels) + self.beta * torch.clamp(margin + E_id - E_ood, min=0)

          return loss, id_loss

    def validation_step(self, batch, device, temperature, metric_dict):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)  #[b]

        # # init ground truth, pred prob vairables
        # gts = [] # ground truth labels
        # probs = [] # predicted prob
        # diff = [0, 1] # accummulated differences 

        # redefining auroc, auprin and auprout

        out = self(images)                    # Generate predictions # [b, n_class]

        # generating prediction results for id samples
        mask = [labels[example].item() > 0 for example in range(labels.shape[0])]
        id_out = out[mask]
        id_labels = labels[mask]

        loss = F.cross_entropy(id_out, id_labels)   # Calculate loss
        acc = accuracy(id_out, id_labels)           # Calculate accuracy

        pred_prob = F.softmax(out/temperature, dim = 1)   # softmaxing # [b, n_class]
        log_pred_prob = F.log_softmax(out/temperature, dim = 1) #[b, n_class]
        # compute score
        
        av_entropy = torch.sum(log_pred_prob * pred_prob, dim=1) #averaged across all classifiers [b]
        max_pred_prob = torch.max(pred_prob, dim=1).values # [b]
        score = max_pred_prob + av_entropy # [b] higher the better for ID

        binary_label_in = torch.Tensor(mask)
        for metric in metric_dict:
            if metric == 'auprout':
                metric_dict[metric].update(score, 
                                           torch.Tensor([not boolean for boolean in mask], # inverted binary mask
                                                         ))
            else:
                metric_dict[metric].update(score, binary_label_in)

        # is_best = auroc > self.best_prec1
        # self.best_prec1 = max(auroc, self.best_prec1)

        # self.save_checkpoint({
        #     'epoch': epoch,
        #     'state_dict': self.model.state_dict(),
        #     'best_prec1': self.best_prec1,
        #     'opt' : self.opt.state_dict(),
        # }, is_best, checkpoint=self.args.checkpoint)

        # calculate accuracy for ID data

        return {'val_loss': loss.detach(), 'val_acc': acc, 'metric_dict': metric_dict}
    

class GoogleNetModel(BaseModel):
    def __init__(self, num_classes, beta=0.5):
        super().__init__()
        self.network = googlenet(pretrained=True)
        self.num_classes = num_classes
        self.beta = beta

        # # Freeze all the parameters of the model
        # for param in self.network.parameters():
        #     param.requires_grad = False

        # # Unfreeze the last three layers
        # for param in list(self.network.parameters())[-3:]:
        #     param.requires_grad = True

        # # Find the last linear layer in the classifier
        # last_linear_layer = None
        # for layer in reversed(self.network.classifier):
        #     if isinstance(layer, nn.Linear):
        #         last_linear_layer = layer
        #         break

        # # Check if a linear layer was found
        # num_ftrs = self.network.fc.in_features
        # if last_linear_layer is not None:
        #     num_ftrs = last_linear_layer.in_features
        # else:
        #     raise ValueError("No linear layer found in classifier")

        # Add custom layers
        num_ftrs = self.network.fc.in_features
        self.network.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

    def training_step(self, batch, batch_ood, device, mode = "OOD", margin = 0.4):

        """
        mode can be "normal" or "OOD"
        run inference, generate losses, output losses
        """

        # generate prediction first, normal image and labels to device
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        images_ood, labels_ood = batch_ood
        images_ood = images_ood.to(device)
        labels_ood = labels_ood.to(device)

        if mode == "normal":
          out = self(images)                  #[bs]
          loss = F.cross_entropy(out, labels) # Calculate loss

          return loss

        elif mode == "OOD":

          """
          requires running on normal inference and OE MIX
          """

          ##### ID LOSS #######
          out_id = self(images) # [b, num_class]
          id_loss = F.cross_entropy(out_id, labels)

          ### overall loss ######
          # ref: https://github.com/YU1ut/Ensemble-of-Leave-out-Classifiers/blob/master/eloc_solver.py#L63

          # somehow F.log_softmax(out_id, dim=1) * F.softmax(out_id, dim=1) this is entropy

          E_id = -torch.mean(torch.sum(F.log_softmax(out_id, dim=1) * F.softmax(out_id, dim=1), dim=1))

          output_ood = self(images_ood) # [b, num_class]

          E_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))

          loss = F.cross_entropy(out_id, labels) + self.beta * torch.clamp(margin + E_id - E_ood, min=0)

          return loss, id_loss

    def validation_step(self, batch, device, temperature, metric_dict):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)  #[b]

        # # init ground truth, pred prob vairables
        # gts = [] # ground truth labels
        # probs = [] # predicted prob
        # diff = [0, 1] # accummulated differences 

        # redefining auroc, auprin and auprout

        out = self(images)                    # Generate predictions # [b, n_class]

        # generating prediction results for id samples
        mask = [labels[example].item() > 0 for example in range(labels.shape[0])]
        id_out = out[mask]
        id_labels = labels[mask]

        loss = F.cross_entropy(id_out, id_labels)   # Calculate loss
        acc = accuracy(id_out, id_labels)           # Calculate accuracy

        pred_prob = F.softmax(out/temperature, dim = 1)   # softmaxing # [b, n_class]
        log_pred_prob = F.log_softmax(out/temperature, dim = 1) #[b, n_class]
        # compute score
        
        av_entropy = torch.sum(log_pred_prob * pred_prob, dim=1) #averaged across all classifiers [b]
        max_pred_prob = torch.max(pred_prob, dim=1).values # [b]
        score = max_pred_prob + av_entropy # [b] higher the better for ID

        binary_label_in = torch.Tensor(mask)
        for metric in metric_dict:
            if metric == 'auprout':
                metric_dict[metric].update(score, 
                                           torch.Tensor([not boolean for boolean in mask], # inverted binary mask
                                                         ))
            else:
                metric_dict[metric].update(score, binary_label_in)

        # is_best = auroc > self.best_prec1
        # self.best_prec1 = max(auroc, self.best_prec1)

        # self.save_checkpoint({
        #     'epoch': epoch,
        #     'state_dict': self.model.state_dict(),
        #     'best_prec1': self.best_prec1,
        #     'opt' : self.opt.state_dict(),
        # }, is_best, checkpoint=self.args.checkpoint)

        # calculate accuracy for ID data

        return {'val_loss': loss.detach(), 'val_acc': acc, 'metric_dict': metric_dict}


class MobileNetModel(BaseModel):
    def __init__(self, num_classes, beta=0.5):
        super().__init__()
        self.network = mobilenet_v2(pretrained=True)
        self.num_classes = num_classes
        self.beta = beta

        # # Freeze all the parameters of the model
        # for param in self.network.parameters():
        #     param.requires_grad = False

        # # Unfreeze the last three layers
        # for param in list(self.network.parameters())[-3:]:
        #     param.requires_grad = True

        # Find the last linear layer in the classifier
        last_linear_layer = None
        for layer in reversed(self.network.classifier):
            if isinstance(layer, nn.Linear):
                last_linear_layer = layer
                break

        # Check if a linear layer was found
        num_ftrs = self.network.classifier[1].in_features
        if last_linear_layer is not None:
            num_ftrs = last_linear_layer.in_features
        else:
            raise ValueError("No linear layer found in classifier")

        # Add custom layers
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

    def training_step(self, batch, batch_ood, device, mode = "OOD", margin = 0.4):

        """
        mode can be "normal" or "OOD"
        run inference, generate losses, output losses
        """

        # generate prediction first, normal image and labels to device
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        images_ood, labels_ood = batch_ood
        images_ood = images_ood.to(device)
        labels_ood = labels_ood.to(device)

        if mode == "normal":
          out = self(images)                  #[bs]
          loss = F.cross_entropy(out, labels) # Calculate loss

          return loss

        elif mode == "OOD":

          """
          requires running on normal inference and OE MIX
          """

          ##### ID LOSS #######
          out_id = self(images) # [b, num_class]
          id_loss = F.cross_entropy(out_id, labels)

          ### overall loss ######
          # ref: https://github.com/YU1ut/Ensemble-of-Leave-out-Classifiers/blob/master/eloc_solver.py#L63

          # somehow F.log_softmax(out_id, dim=1) * F.softmax(out_id, dim=1) this is entropy

          E_id = -torch.mean(torch.sum(F.log_softmax(out_id, dim=1) * F.softmax(out_id, dim=1), dim=1))

          output_ood = self(images_ood) # [b, num_class]

          E_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))

          loss = F.cross_entropy(out_id, labels) + self.beta * torch.clamp(margin + E_id - E_ood, min=0)

          return loss, id_loss

    def validation_step(self, batch, device, temperature, metric_dict):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)  #[b]

        # # init ground truth, pred prob vairables
        # gts = [] # ground truth labels
        # probs = [] # predicted prob
        # diff = [0, 1] # accummulated differences 

        # redefining auroc, auprin and auprout

        out = self(images)                    # Generate predictions # [b, n_class]

        # generating prediction results for id samples
        mask = [labels[example].item() > 0 for example in range(labels.shape[0])]
        id_out = out[mask]
        id_labels = labels[mask]

        loss = F.cross_entropy(id_out, id_labels)   # Calculate loss
        acc = accuracy(id_out, id_labels)           # Calculate accuracy

        pred_prob = F.softmax(out/temperature, dim = 1)   # softmaxing # [b, n_class]
        log_pred_prob = F.log_softmax(out/temperature, dim = 1) #[b, n_class]
        # compute score
        
        av_entropy = torch.sum(log_pred_prob * pred_prob, dim=1) #averaged across all classifiers [b]
        max_pred_prob = torch.max(pred_prob, dim=1).values # [b]
        score = max_pred_prob + av_entropy # [b] higher the better for ID

        binary_label_in = torch.Tensor(mask)
        for metric in metric_dict:
            if metric == 'auprout':
                metric_dict[metric].update(score, 
                                           torch.Tensor([not boolean for boolean in mask], # inverted binary mask
                                                         ))
            else:
                metric_dict[metric].update(score, binary_label_in)

        # is_best = auroc > self.best_prec1
        # self.best_prec1 = max(auroc, self.best_prec1)

        # self.save_checkpoint({
        #     'epoch': epoch,
        #     'state_dict': self.model.state_dict(),
        #     'best_prec1': self.best_prec1,
        #     'opt' : self.opt.state_dict(),
        # }, is_best, checkpoint=self.args.checkpoint)

        # calculate accuracy for ID data

        return {'val_loss': loss.detach(), 'val_acc': acc, 'metric_dict': metric_dict}
