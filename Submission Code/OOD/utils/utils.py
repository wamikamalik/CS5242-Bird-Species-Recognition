#### utils ####
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import torch
from torcheval.metrics.functional import binary_auprc

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(vutils.make_grid(images, nrow=8).permute(1, 2, 0))
        break

######### to add in training loop #####
## avg functio

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# # soft CE for soft labels
# class SoftCE(nn.Module):
#     def __init__(self, reduction="mean"):
#         super().__init__()
#         self.reduction = reduction

#     def forward(self, logits, soft_targets):
#         preds = logits.log_softmax(dim=-1)
#         assert preds.shape == soft_targets.shape

#         loss = torch.sum(-soft_targets * preds, dim=-1)

#         if self.reduction == "mean":
#             return torch.mean(loss)
#         elif self.reduction == "sum":
#             return torch.sum(loss)
#         elif self.reduction == "none":
#             return loss
#         else:
#             raise ValueError("Reduction type '{:s}' is not supported!".format(self.reduction))


# # author of mixoe defined accuracy
# def accuracy_author(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size).item())
#         return res

##### include extra validation calculations here
diff = [0, 1]
def Tpr95(X1, Y1):
    #calculate the falsepositive error when tpr is 95%
    total = 0.0
    fpr = 0.0
    for delta in diff:
        delta = float(delta)
        tpr = np.sum(np.sum(X1 >= delta)) / float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr/total

    return fprBase

def auprIn(X1, Y1):
    #calculate the AUPR
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff:
        delta = float(delta)
        tp = np.sum(np.sum(X1 >= delta)) / float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def auprOut(X1, Y1):
    #calculate the AUPR
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff[::-1]:
        delta = float(delta)
        fp = np.sum(np.sum(X1 < delta)) / float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision
        
    return auprBase

class AP():
    def __init__(self):
        self.inputs = None
        self.labels = None
    def update(self, inputs, labels):
        if self.inputs == None:
            self.inputs = inputs
        if self.labels == None:
            self.labels = labels
        else:
            self.inputs = torch.concat((self.inputs, inputs),0)
            self.labels = torch.concat((self.labels, labels),0)
    def compute(self, device, mode = "in"):
        print(self.inputs, self.labels)
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        if mode == "out":
            aupr = binary_auprc(-self.inputs, self.labels)
        else:
            aupr = binary_auprc(self.inputs, self.labels)
        return aupr