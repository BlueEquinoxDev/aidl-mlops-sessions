import torch
import torch.utils
import torch.utils.data
from torchvision import transforms

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model, save_optimizer

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import os

# VARIABLES
OUTPUT_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/output"
MODEL_NAME = "cmnist_model_hyp_tune"
PLOT_NAME = "loss-acc-curves_hyp_tune"
IMAGES_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/archive/data/data"
LABELS_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/archive/chinese_mnist.csv"


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def train_single_epoch(
        train_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        epoch: int,
        log_interval: int,
        ) -> Tuple[float, float]:
    
    # Activate the train=True flag inside the model
    network.train()

    train_loss = []
    acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        """
        print(f"Data Shape: {data.shape}")
        print(f"Target Shape: {target.shape}")
        # Show the data
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(data), size=(1,)).item()
            img = data[sample_idx]
            label = target[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()
        """
                
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Compute metrics
        acc += accuracy(outputs=output, labels=target)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader.dataset)
    print('\nTrain Epoch: {}, Length: {}, Final Accuracy: {:.0f}%'.format(epoch, len(train_loader.dataset), avg_acc))

    return np.mean(train_loss), avg_acc

    
@torch.no_grad() # decorator: avoid computing gradients
def eval_single_epoch(
        test_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        criterion: torch.nn.functional,
        ) -> Tuple[float, float]:
    
    # Dectivate the train=True flag inside the model
    network.eval()

    test_loss = 0
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = network(data)

        # Apply the loss criterion and accumulate the loss
        test_loss += criterion(output, target).item()

        # compute number of correct predictions in the batch
        acc += accuracy(outputs=output, labels=target)

    test_loss /= len(test_loader)
    # Average accuracy across all correct predictions batches now
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc
        
    

def train_model(config):
    
    # Prepare a set of transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5), (0.5))
    ])

    my_dataset = MyDataset(images_path=IMAGES_PATH, labels_path=LABELS_PATH, transform=transform)

    # Split the data
    generator1 = torch.Generator().manual_seed(123)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[10000, 2500, 2500], generator=generator1)
    print(f"Length of train_dataset={len(train_dataset)}, val_dataset={len(val_dataset)}, test_dataset={len(test_dataset)}")

    """
    # Show the data
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    """

    # Generate the Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=6, pin_memory=True)

    # Load model
    my_model = MyModel(dropout=config['dropout'])
    if torch.mps.device_count()>1:
        my_model = torch.nn.DataParallel(my_model)
    my_model.to(device)

    # Create an optimizer to compute the model
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config['learning_rate'])

    # Set a criterion valid for multiclass classification
    criterion = torch.nn.NLLLoss(reduction='mean') #F.cross_entropy

    # Evaluate model accuracy at the begining:
    """
    print("Model without training")
    print("="*10)
    te_loss, te_acc = eval_single_epoch(test_loader=test_loader,
                                            network=my_model,
                                            criterion=criterion)
    """

    # Record a losses and acc
    tr_losses = []
    tr_accs = []
    te_losses = []
    te_accs = []


    for epoch in range(config["epochs"]):
        tr_loss, tr_acc = train_single_epoch(train_loader=train_loader,
                                             network=my_model, 
                                             optimizer=optimizer, 
                                             criterion=criterion, 
                                             log_interval=config['log_interval'], 
                                             epoch=epoch)
        te_loss, te_acc = eval_single_epoch(test_loader=test_loader,
                                            network=my_model,
                                            criterion=criterion)
        
        te_losses.append(te_loss)
        te_accs.append(te_acc)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)

        # Save the model during execution
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        output_model_path = os.path.join(OUTPUT_PATH, MODEL_NAME)
        save_model(my_model, path=output_model_path)
        output_optimizer_path = os.path.join(OUTPUT_PATH, "optimizer")
        save_optimizer(optimizer, path=output_optimizer_path)
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(tr_losses, label='train')
        plt.plot(te_losses, label='eval')
        plt.legend()
        plt.subplot(2,1,2)
        plt.xlabel('Epoch')
        plt.ylabel('Eval Accuracy [%]')
        plt.plot(tr_accs, label='train')
        plt.plot(te_accs, label='eval')
        plt.legend()

        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        output_plot_path = os.path.join(OUTPUT_PATH, PLOT_NAME + ".png")
        plt.savefig(output_plot_path)


    return my_model


if __name__ == "__main__":

    config = {
        'batch_size': 64,
        'epochs': 10,
        'learning_rate': 0.001,
        'dropout': 0.5,
        'log_interval': 10,
    }
    result_model = train_model(config)

    
