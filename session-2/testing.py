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
BEST_MODEL_PATH = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/output/cmnist_model"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

@torch.no_grad() # decorator: avoid computing gradients
def test_model(config: Dict, checkpoint_path):
    
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
    # Input your best model to test it
    config={
        "act_func": "leakyrelu",
        "batch_size": 32.29895820878754,
        "dropout": 0.2770690579784582,
        "epochs": 10,
        "kernel": 5,
        "learning_rate": 0.0005587727084518315,
        "log_interval": 10,
        "neurons": 256
    }
    
    model_name = "train_model_a52b173c"
    checkpoint_path = "/Users/Rafel.Febrer/Data Science - Rafel/Postgrau DeepLearning/MLOps/MLOpsLAB/LAB1-2/aidl-2025-winter-mlops/session-2/ray/train_model_2024-11-17_18-50-19/train_model_a52b173c_12_act_func=leakyrelu,batch_size=32.2990,dropout=0.2771,epochs=10,kernel=5,learning_rate=0.0006,log_interval=_2024-11-17_18-50-33/checkpoint_000009/checkpoint.pt"

    print("="*30)
    print("TEST DATASET")
    # Run the best config
    print(test_model(config=config, checkpoint_path=checkpoint_path))
    print("="*30)