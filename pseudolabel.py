import torch
import torch.nn as nn
from tqdm import tqdm

from tools import prepare, mixup, preprocess, getCorrects

class AlphaWeight:
    def __init__(self, T1, T2, af):
        self.T1 = T1
        self.T2 = T2
        self.af = af

    def __call__(self, step):
        if step < self.T1:
            return 0.0
        elif step > self.T2:
            return self.af
        else:
             return ((step - self.T1) / (self.T2 - self.T1)) * self.af

def pl_train_epoch(model, epoch, criterion, optimizer, scheduler, dataloader, dataloader_u, device, step, every_n, no_mixup, alpha_weight, nfft, num_classes):
    model.train()
        
    num_batches = len(dataloader_u)
    num_samples = 0
    
    running_loss = 0.0
    running_corrects = 0
    
    for batch, (inputs_u, _) in enumerate(tqdm(dataloader_u)):
        # Transfer Data to GPU if available
        inputs_u = inputs_u.to(device)
        if not no_mixup:
            pass
            #inputs_u, _ = mixup(inputs_u, torch.arange(end=len(inputs_u)), min_seq=1, max_seq=3)
        inputs_u = preprocess(inputs_u, nfft)
        
        # Forward Pass to get the pseudo labels
        #model.eval()
        outputs_unlabeled = model(inputs_u)
        _, pseudo_labels = torch.max(outputs_unlabeled, 1)
        pseudo_labels = torch.nn.functional.one_hot(pseudo_labels, num_classes)
        #pseudo_labels = outputs_unlabeled.detach().clone()
        #pseudo_labels[pseudo_labels >= 0.5] = 1.0
        #pseudo_labels[pseudo_labels < 0.5] = 0.0
        #model.train()
            
        # Now calculate the unlabeled loss using the pseudo label
        #outputs = model(inputs_u)
        unlabeled_loss = alpha_weight(step) * criterion(outputs_unlabeled, pseudo_labels)   

        # Backpropogate
        optimizer.zero_grad()
        unlabeled_loss.backward()
        optimizer.step()
        
        # For every x batches train one epoch on labeled data 
        if batch % every_n == 0:

            # Normal training procedure
            for batch, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                if not no_mixup:
                    inputs, labels = mixup(inputs, labels, min_seq=1, max_seq=3)
                inputs = preprocess(inputs, nfft)
                
                outputs = model(inputs)
                labeled_loss = criterion(outputs, labels)

                optimizer.zero_grad()
                labeled_loss.backward()
                optimizer.step()
        
                # Calculate Loss
                running_loss += labeled_loss.item() * inputs.size(0)
                running_corrects += getCorrects(outputs, labels)
                num_samples += inputs.size(0)

            # Now we increment step by 1
            step += 1
    
        # Perform learning rate step
        if scheduler is not None:
            scheduler.step(epoch + batch / num_batches)
            
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects / num_samples
    
    return epoch_loss, epoch_acc, step