from tqdm import tqdm
from tools import prepare, mixup, preprocess

def byol_train_epoch(model, epoch, criterion, optimizer, scheduler, dataloader, device, no_mixup, nfft):
    model.train()
    
    running_loss = 0.0
    
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    
    for batch, (inputs, labels) in enumerate(tqdm(dataloader)):
        # Transfer Data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)
        if not no_mixup:
            inputs, _ = mixup(inputs, labels, min_seq=1, max_seq=3, p_min=0.3)
        inputs = preprocess(inputs, nfft).unsqueeze(1)
         
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward Pass
        loss = model(inputs)
        
        # Calculate gradients
        loss.backward()
        
        # Update Weights
        optimizer.step()
        model.update_moving_average()
        
        # Calculate Loss
        running_loss += loss.item() * inputs.size(0)
    
        # Perform learning rate step
        #scheduler.step(epoch + batch / num_batches)
            
    epoch_loss = running_loss / num_samples
    
    return epoch_loss