import ray.train
import ray.tune
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
from ray import tune
import ray
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
import tempfile

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict
import os

# VARIABLES
OUTPUT_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/output"
MODEL_NAME = "cmnist_model_hyp_tune"
PLOT_NAME = "loss-acc-curves_hyp_tune"
IMAGES_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/archive/data/data"
LABELS_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/archive/chinese_mnist.csv"
RAY_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/ray"

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
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))

    return test_loss, test_acc



def train_model(config):

    # Prepare a set of transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])

    my_dataset = MyDataset(images_path=IMAGES_PATH, labels_path=LABELS_PATH, transform=transform)

    # Split the data
    generator1 = torch.Generator().manual_seed(123)
    train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset=my_dataset, lengths=[10000, 2500, 2500], generator=generator1)
    print(f"Length of train_dataset={len(train_dataset)}, val_dataset={len(val_dataset)}")

    # Generate the Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(config['batch_size']), shuffle=True)

    # Load model
    my_model = MyModel(dropout=config['dropout'], kernel=config['kernel'], neurons=config['neurons'], act_func=config['act_func'])
    if torch.mps.device_count()>1:
        my_model = torch.nn.DataParallel(my_model)
    my_model.to(device)

    # Create an optimizer to compute the model
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config['learning_rate'])

    # Set a criterion valid for multiclass classification
    criterion = torch.nn.NLLLoss(reduction='mean')

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
        te_loss, te_acc = eval_single_epoch(test_loader=val_loader,
                                            network=my_model,
                                            criterion=criterion)
        
        te_losses.append(te_loss)
        te_accs.append(te_acc)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)

        """
        # Save the model during execution
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        output_model_path = os.path.join(OUTPUT_PATH, MODEL_NAME)
        save_model(my_model, path=output_model_path)
        """

        #ray.train.report({"val_loss": te_loss})

        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": my_model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            ray.train.report(metrics={"val_loss": te_loss}, checkpoint=ray.train.Checkpoint.from_directory(tempdir))

    #return my_model


@torch.no_grad() # decorator: avoid computing gradients
def test_model(config: Dict, checkpoint):
    
    # Prepare a set of transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])

    my_dataset = MyDataset(images_path=IMAGES_PATH, labels_path=LABELS_PATH, transform=transform)

    # Split the data
    generator1 = torch.Generator().manual_seed(123)
    _, _, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[10000, 2500, 2500], generator=generator1)
    
    # Load test DataLoader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=True)

    # Load model
    my_model = MyModel(dropout=config['dropout'], kernel=config['kernel'], neurons=config['neurons'], act_func=config['act_func'])
    if torch.mps.device_count()>1:
        my_model = torch.nn.DataParallel(my_model)
    my_model.to(device)

    # Load back training state
    checkpoint_path = os.path.join(checkpoint.path, "checkpoint.pt")
    model_state = torch.load(checkpoint_path, weights_only=True)
    my_model.load_state_dict(model_state["model_state"])

    # Dectivate the train=True flag inside the model
    my_model.eval()

    # Set a criterion valid for multiclass classification
    criterion = torch.nn.NLLLoss(reduction='mean')

    acc = 0
    test_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = my_model(data)

        # Apply the loss criterion and accumulate the loss
        test_loss += criterion(output, target).item()

        # compute number of correct predictions in the batch
        acc += accuracy(outputs=output, labels=target)

    # Average accuracy across all correct predictions batches now
    test_acc = 100. * acc / len(test_loader.dataset)
    test_loss /= len(test_loader)
    
    return '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        )


if __name__ == "__main__":
    # Create the search space
    search_space = {
            'learning_rate': tune.loguniform(1e-4, 5e-3),
            'batch_size': tune.uniform(20, 70), #tune.qrandint(20, 70),
            'epochs': 10,
            'dropout': tune.uniform(0.2, 0.6),
            'log_interval': 10,
            'act_func': tune.choice(["relu", "leakyrelu", "tanh"]),
            'kernel': tune.choice([3, 4, 5]),
            'neurons': tune.choice([512, 256, 128]),
    }

    # Define hyperopt search model
    hyperoptsearch = HyperOptSearch(random_state_seed=123)
    #bayesoptsearch = BayesOptSearch()

    # Init ray
    ray.init(configure_logging=False)
    analysis = tune.run(
        run_or_experiment=train_model,
        resources_per_trial={"cpu": 1},
        config=search_space,
        search_alg=hyperoptsearch, #bayesoptsearch, # hyperoptsearch
        storage_path=RAY_PATH,
        metric="val_loss",
        mode="min",
        num_samples=50,
    )

    # Share best model
    #print("Best hyperparameters found were: ", analysis.best_config)
    print("Best hyperparameters found were: ", analysis.best_config)
    print(analysis.best_trial)
    
    

    print("="*30)
    print("TEST DATASET")
    # Run the best config
    print(test_model(config=analysis.best_config, checkpoint=analysis.get_best_checkpoint(trial=analysis.best_trial)))
    print("="*30)
