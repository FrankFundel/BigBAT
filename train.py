import argparse
import json
import wandb
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, multilabel_confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from byol_pytorch import BYOL

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T

from ASL import AsymmetricLoss
from BigBAT import BigBAT
from tools import prepare, mixup, preprocess, noise, getCorrects, RandomApply
from byol import byol_train_epoch
from fixmatch import consistency_loss, fixmatch_train_epoch
from pseudolabel import AlphaWeight, pl_train_epoch

# Function to load classes from JSON file
def load_classes_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        classes_list = json.load(file)
    classes = { classes_list[i]: i for i in range(len(classes_list)) }
    return classes_list, classes

# Create argument parser
parser = argparse.ArgumentParser(description="Train BigBAT easily on custom datasets.")
parser.add_argument('json_file', type=str, help="Path to the JSON file containing class names (array).")
parser.add_argument('--nfft', type=int, default=512, help="Number of FFT points")
parser.add_argument('--max_len', type=int, default=60, help="Maximum length")
parser.add_argument('--patch_len', type=int, default=44, help="Patch length")
parser.add_argument('--patch_skip', type=int, default=22, help="Patch skip length")
parser.add_argument('--max_seqs', type=int, default=None, help="Maximum sequences")
parser.add_argument('--min_seqs', type=int, default=None, help="Minimum sequences")
parser.add_argument('data_path', type=str, help="Path to the data (.h5) file")
parser.add_argument('--holdout', type=float, default=0, help="Proportion of data to hold out (0 to 1)")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--epochs', type=int, default=15, help="Number of epochs")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--warmup_epochs', type=int, default=3, help="Number of warm-up epochs")
parser.add_argument('--wandb_project', type=str, default="BigBAT-UFS", help="Weights & Biases project name")
parser.add_argument('--wandb_entity', type=str, default="frankfundel", help="Weights & Biases entity name")
parser.add_argument('--model_filename', type=str, default='BigBAT.pth', help="Filename for the saved model")
parser.add_argument('--repeats', type=int, default=5, help="Number of repeats for evaluation")
parser.add_argument('--figure_filename', type=str, default='confusion_matrix.png', help="Filename for saving the confusion matrix figure")
parser.add_argument('--method', type=str, default='standard', choices=['standard', 'BYOL', 'FixMatch', 'PseudoLabel'], help="The method of training. Needs to be done in combination of --holdout > 0")
parser.add_argument('--T1', type=int, default=1, help="At what epoch unlabeled data is introduce.?")
parser.add_argument('--T2', type=int, default=6, help="At what epoch unlabeled is unintroduced.")
parser.add_argument('--every_n', type=int, default=10, help="Every n batches, use unlabled data.")
parser.add_argument('--no_mixup', action='store_true', help="If mixup should not be used.")
parser.add_argument('--lambda_u', type=float, default=1.0, help="Weight of unlabeled loss in FixMatch.")

# Parse arguments
args = parser.parse_args()
nfft = args.nfft
max_len = args.max_len
patch_len = args.patch_len
patch_skip = args.patch_skip
max_seqs = args.max_seqs
min_seqs = args.min_seqs
data_path = args.data_path
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
warmup_epochs = args.warmup_epochs
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
model_filename = args.model_filename
repeats = args.repeats
figure_filename = args.figure_filename
method = args.method
holdout = args.holdout
no_mixup = args.no_mixup

# Initializing
classes_list, classes = load_classes_from_json(args.json_file)

num_bands = nfft // 2 + 1
samples_per_step = patch_skip * (nfft // 4)
seq_len = (max_len + 1) * samples_per_step
seq_skip = seq_len // 4

print("Loading data...")
X_train, Y_train, X_test, Y_test, X_val, Y_val = prepare(data_path, classes, seq_len, seq_skip, max_seqs, min_seqs)

print("Total sequences:", len(X_train) + len(X_test) + len(X_val))
print("Train sequences:", X_train.shape, Y_train.shape)
print("Test sequences:", X_test.shape, Y_test.shape)
print("Validation sequences:", X_val.shape, Y_val.shape)

# Holdout for unsupervised training
if holdout > 0:
    print("Using holdout of " + str(holdout))
    h_train = int(len(X_train) * holdout)
    X_unlabeled = X_train[:h_train]
    X_train = X_train[h_train:]
    Y_train = Y_train[h_train:]

# Define device and model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

patch_embedding = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(2, 3), padding=3),
    nn.BatchNorm2d(16),
    nn.ReLU(),

    nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(2, 3), padding=3),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 3), padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 3), padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),

    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
).to(device)

num_classes = len(classes_list)
d_model = 64
nhead = 2
dim_feedforward = 32
num_layers = 2
dropout = 0.3
classifier_dropout = 0.3

model = BigBAT(
    max_len=max_len,
    patch_len=patch_len,
    patch_skip=patch_skip,
    d_model=d_model,
    num_classes=num_classes,
    patch_embedding=patch_embedding,
    use_cls=True,
    nhead=2,
    dim_feedforward=dim_feedforward,
    num_layers=num_layers,
    dropout=dropout,
    classifier_dropout=classifier_dropout,
    squeeze_first=(method == 'BYOL')
)

preprocessed_shape = preprocess(X_train[0].unsqueeze(0), nfft).shape # (1, 1343, 257)

if method == 'standard':
    pass
elif method == 'BYOL':
    custom_augment_fn = nn.Sequential(
        RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomApply(T.GaussianBlur((23, 23), (1.5, 1.5)), p=0.3),
        RandomApply(noise, p=0.5),
        T.RandomErasing()
        #T.RandomResizedCrop(preprocessed_shape[1:]) could be used for time shift augmentation
    )

    byol_model = BYOL(
        model,
        image_size = preprocessed_shape,
        hidden_layer = 'classifier',
        projection_size = 32,
        projection_hidden_size = 64,
        augment_fn = custom_augment_fn
    )
    byol_model.to(device)
elif method == 'FixMatch':
    augment_s = nn.Sequential(
        #RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomApply(T.GaussianBlur((23, 23), (1.5, 1.5)), p=0.3),
        RandomApply(noise, p=0.5),
        T.RandomErasing()
        #T.RandomResizedCrop(preprocessed_shape[1:]) could be used for time shift augmentation
    )
    
    augment_w = nn.Sequential(
        #RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        #T.RandomHorizontalFlip(),
        #T.RandomVerticalFlip(),
        RandomApply(T.GaussianBlur((23, 23), (1.5, 1.5)), p=0.3),
        RandomApply(noise, p=0.5)
        #T.RandomErasing()
        #T.RandomResizedCrop(preprocessed_shape[1:]) could be used for time shift augmentation
    )

    lambda_u = args.lambda_u
elif method == 'PseudoLabel':
    T1 = args.T1
    T2 = args.T2
    af = 1
    alpha_weight = AlphaWeight(T1, T2, af)
    step = 0
    every_n = args.every_n

model.to(device)

print("Model loaded!")

# Create dataloaders
train_len = batch_size * int(len(X_train) / batch_size)
test_len = batch_size * int(len(X_test) / batch_size)
val_len = batch_size * int(len(X_val) / batch_size)

train_data = TensorDataset(X_train[:train_len], Y_train[:train_len])
test_data = TensorDataset(X_test[:test_len], Y_test[:test_len])
val_data = TensorDataset(X_val[:val_len], Y_val[:val_len])

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

unlabeled_len = batch_size * int(len(X_unlabeled) / batch_size)
unlabeled_data = TensorDataset(X_unlabeled[:unlabeled_len], torch.zeros(unlabeled_len))
unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size)
print("Created dataloaders!")

# Initialize training
criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True, dampening=0)
if warmup_epochs > 0:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=warmup_epochs)
else:
    scheduler = None

min_val_loss = torch.inf

torch.autograd.set_detect_anomaly(True)

def train_epoch(model, epoch, criterion, optimizer, scheduler, dataloader, device):
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    
    for batch, (inputs, labels) in enumerate(tqdm(dataloader)):
        # Transfer Data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)
        if not no_mixup:
            inputs, labels = mixup(inputs, labels, min_seq=1, max_seq=3, p_min=0.3)
        inputs = preprocess(inputs, nfft)
         
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(inputs)

        # Compute Loss (ASL)
        loss = criterion(outputs, labels)
        
        # Calculate gradients
        loss.backward()
        
        # Update Weights
        optimizer.step()
        
        # Calculate Loss
        running_loss += loss.item() * inputs.size(0)
        running_corrects += getCorrects(outputs, labels)
    
        # Perform learning rate step
        if scheduler is not None:
            scheduler.step(epoch + batch / num_batches)
            
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects / num_samples
    
    return epoch_loss, epoch_acc

def test_epoch(model, epoch, criterion, optimizer, dataloader, device):
    model.eval()
    
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        for batch, (inputs, labels) in enumerate(tqdm(dataloader)):
            # Transfer Data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)
            if not no_mixup:
                inputs, labels = mixup(inputs, labels, min_seq=1, max_seq=3)
            inputs = preprocess(inputs)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(inputs)

            # Compute Loss
            loss = criterion(outputs, labels)

            # Calculate Loss
            running_loss += loss.item() * inputs.size(0)
            running_corrects += getCorrects(outputs, labels)

        epoch_loss = running_loss / num_samples
        epoch_acc = running_corrects / num_samples
    
    return epoch_loss, epoch_acc

# Initizalize weights and biases
if wandb_project is not None and wandb_entity is not None:
    wandb_config = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "d_model": d_model,
        "nhead": nhead,
        "dim_feedforward": dim_feedforward,
        "num_layers": num_layers,
        "dropout": dropout,
        "classifier_dropout": classifier_dropout,
        "method": method,
    }
    wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config)

# Start training

for epoch in range(epochs):
    end = time.time()
    print(f"==================== Starting at epoch {epoch} ====================", flush=True)

    if method == 'BYOL':
        train_loss = byol_train_epoch(byol_model, epoch, criterion, optimizer, scheduler, train_loader, device, no_mixup, nfft)
        train_acc = 0.0
    elif method == 'FixMatch':
        train_loss, train_acc = fixmatch_train_epoch(model, epoch, criterion, optimizer, scheduler, train_loader, unlabeled_loader, device, augment_w, augment_s, no_mixup, lambda_u, nfft)
    elif method == 'PseudoLabel':
        train_loss, train_acc, step = pl_train_epoch(model, epoch, criterion, optimizer, scheduler, train_loader, unlabeled_loader, device, step, every_n, no_mixup, alpha_weight, nfft, num_classes)
    else:
        train_loss, train_acc = train_epoch(model, epoch, criterion, optimizer, scheduler, train_loader, device)
    
    print('Training loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc), flush=True)
    
    val_loss, val_acc = test_epoch(model, epoch, criterion, optimizer, val_loader, device)
    print('Validation loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc), flush=True)
    
    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })
    
    if min_val_loss > val_loss:
        print('val_loss decreased, saving model', flush=True)
        min_val_loss = val_loss
        
        # Saving State Dict
        torch.save(model.state_dict(), model_filename)

# Load after training
model.load_state_dict(torch.load(model_filename))

print("Starting evaluation!")

# Evaluation
mixed_corrects = 0.0
predictions = []
targets = []
for r in range(repeats):
    # iterate over test data
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = mixup(inputs, labels, min_seq=1, max_seq=3)
        inputs = preprocess(inputs, nfft)

        output = model(inputs) # Feed Network
        predictions.extend(output.data.cpu().numpy())
        targets.extend(labels.data.cpu().numpy())
        mixed_corrects += getCorrects(output, labels)

# Metrics calculation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

mixed_test_acc = mixed_corrects / (repeats * len(test_data))
mixed_f1_micro = f1_score(sigmoid(np.asarray(predictions)) > 0.5, np.asarray(targets), average='micro')
mixed_f1_macro = f1_score(sigmoid(np.asarray(predictions)) > 0.5, np.asarray(targets), average='macro')

print("Mixed test acc:", mixed_test_acc)
print("Mixed f1 micro:", mixed_f1_micro)
print("Mixed f1 macro:", mixed_f1_macro)

# Model parameters count
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params, "params")

# Confusion Matrix
Y_pred = []
Y_true = []

model.eval()
for inputs, labels in tqdm(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    inputs = preprocess(inputs)
    
    output = model(inputs)  # Feed Network
    output = (output > 0.5).data.cpu().numpy()
    Y_pred.extend(output)  # Save Prediction
    labels = labels.data.cpu().numpy()
    Y_true.extend(labels)  # Save Truth

cf_matrix = multilabel_confusion_matrix(Y_true, Y_pred)

# Plotting Confusion Matrix
f, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.ravel()
for i in range(len(classes_list)):
    disp = ConfusionMatrixDisplay(cf_matrix[i] / np.sum(cf_matrix[i]), display_labels=[0, i])
    disp.plot(ax=axes[i], values_format='.4g')
    disp.ax_.set_title(classes_list[i])
    if i<10:
        disp.ax_.set_xlabel('')
    if i%5!=0:
        disp.ax_.set_ylabel('')
    disp.im_.colorbar.remove()

plt.subplots_adjust(wspace=0.10, hspace=0.1)
f.colorbar(disp.im_, ax=axes)
plt.savefig(figure_filename)
plt.show()

# Filter out mixed and calculate single test accuracy and F1-score
Y_pred2, Y_true2 = [], []
for i, y in enumerate(Y_true):
    if y.sum() == 1:
        Y_true2.append(np.argmax(y))
        Y_pred2.append(np.argmax(Y_pred[i]))

Y_pred = Y_pred2
Y_true = Y_true2
corrects = np.equal(Y_pred, Y_true).sum()
single_test_acc = corrects / len(Y_pred)
single_f1 = f1_score(Y_true, Y_pred, average=None).mean()

print("Single test accuracy:", single_test_acc)
print("Single F1-score:", single_f1)

# Logging to wandb
if wandb_project is not None and wandb_entity is not None:
    wandb.log({
        "mixed_test_acc": mixed_test_acc,
        "mixed_f1_micro": mixed_f1_micro,
        "mixed_f1_macro": mixed_f1_macro,
        "single_test_acc": single_test_acc,
        "single_f1_micro": single_f1,
        "num_params": pytorch_total_params,
    })

    wandb.finish()