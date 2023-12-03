import torch
import torch.nn as nn
from tqdm import tqdm

from tools import prepare, mixup, preprocess, getCorrects

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return nn.functional.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = nn.functional.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return 
    
def consistency_loss(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.5, use_hard_labels=True, criterion=None):
    assert name in ['ce', 'L2', 'L2_mask', 'custom_mask']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return nn.functional.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        logits_w = logits_w.detach()
        outputs_pseudo = torch.sigmoid(logits_w)
        max_probs, pseudo_labels = torch.max(outputs_pseudo, 1)
        pseudo_labels = torch.nn.functional.one_hot(pseudo_labels, num_classes).to(torch.float32)
        mask = max_probs.ge(p_cutoff)
        masked_loss = nn.functional.mse_loss(logits_s[mask], pseudo_labels[mask])
        return masked_loss

    elif name == 'custom_mask':
        logits_w = logits_w.detach()
        outputs_pseudo = torch.sigmoid(logits_w)
        max_probs, pseudo_labels = torch.max(outputs_pseudo, 1)
        pseudo_labels = torch.nn.functional.one_hot(pseudo_labels, num_classes).to(torch.float32)
        mask = max_probs.ge(p_cutoff)
        masked_loss = criterion(logits_s[mask], pseudo_labels[mask])
        return masked_loss
    
    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean()

    else:
        assert Exception('Not Implemented consistency_loss')


def fixmatch_train_epoch(model, epoch, criterion, optimizer, scheduler, dataloader, dataloader_u, device, augment_w, augment_s, no_mixup, lambda_u, nfft):
    model.train()
        
    num_batches = len(dataloader_u)
    num_samples = 0
    
    running_loss = 0.0
    running_corrects = 0

    iter_u = iter(dataloader_u)
        
    # Normal training procedure
    for batch, (inputs, labels) in enumerate(tqdm(dataloader)):
        try:
            inputs_u, _ = next(iter_u)
        except StopIteration:
            iter_u = iter(dataloader_u)
            inputs_u, _ = next(iter_u)
        
        inputs_u = inputs_u.to(device)
        inputs_u = preprocess(inputs_u)
        
        inputs, labels = inputs.to(device), labels.to(device)
        if not no_mixup:
            inputs, labels = mixup(inputs, labels, min_seq=1, max_seq=3)
        inputs = preprocess(inputs, nfft)

        # Augment unlabeled data (weakly and strongly)
        inputs_uw = augment_w(inputs_u)
        inputs_us = augment_s(inputs_u)
        
        #outputs_u = model(inputs_u[mask])
        outputs_uw = model(inputs_uw)
        outputs_us = model(inputs_us)
        unlabeled_loss = consistency_loss(outputs_uw, outputs_us, name='L2')
        
        outputs = model(inputs)
        labeled_loss = criterion(outputs, labels) # reduction : mean
        loss = labeled_loss + lambda_u * unlabeled_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate Loss
        running_loss += loss.item() * inputs.size(0)
        running_corrects += getCorrects(outputs, labels)
        num_samples += inputs.size(0)
    
        # Perform learning rate step
        if scheduler is not None:
            scheduler.step(epoch + batch / num_batches)
            
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects / num_samples
    return epoch_loss, epoch_acc