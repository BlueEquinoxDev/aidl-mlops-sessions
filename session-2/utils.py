import torch


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)

    # For others:
    # acc = (preds == labels.view_as(preds)).detach().numpy().sum().item() # .float().detach().numpy().mean()

    # for mac GPU:
    acc = (preds == labels.view_as(preds)).detach().cpu().numpy().sum().item()

    return acc
    

def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_optimizer(optimizer, path):
    torch.save(optimizer.state_dict(), path)