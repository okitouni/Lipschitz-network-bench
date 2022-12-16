# %%
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from monotonenorm import direct_norm, GroupSort
import torch
from tqdm import tqdm
from gradient_descent_the_ultimate_optimizer import gdtuo
from models import get_layer, track_norms
from torchmetrics.functional import accuracy
import wandb
import os

# torch.manual_seed(1)

BATCHSIZE = -1
EPOCHS = 100000
RANDOM_LABELS = True
MODEL = "Lipschitz" # "Lipschitz" or "Unconstrained"
TAU = 256 # rescale temperature of CrossEntropyLoss
MAX_NORM = 2 # max norm of each layer
DATASET = "CIFAR100"
WIDTH = 1024
LR=1e-5
OPTIM="Adam"
TRACK_NORM=False
WANDB = True

name = f"{DATASET}_{MODEL}_{WIDTH}_tau{TAU}_maxnorm{MAX_NORM}"
if RANDOM_LABELS:
    name += "_random"
if WANDB:
    wandb.init(project=f"LipNN", entity="iaifi", name=name)
    wandb.config = {
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCHSIZE,
        "model": "MLP",
        "optimizer": OPTIM,
        "loss": "CrossEntropy",
        "dataset": DATASET,
        "width": WIDTH,
        "depth": 3,
        # "activation": "GroupSort",
    }


norm = direct_norm if MODEL == "Lipschitz" else lambda x, **kwargs: x
# [0, 1] normalization
normalize = transforms.Normalize(
    mean=[x/255.0 for x in [0, 0, 0]], std=[x / 255.0 for x in [1, 1, 1]])
transform = transforms.Compose(
    [transforms.ToTensor(), normalize])
trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
if RANDOM_LABELS:
    trainset.targets = torch.randint(0, 100, (len(trainset),))
bs = BATCHSIZE if BATCHSIZE > 0 else len(trainset)
shuffle = True if BATCHSIZE > 0 else False

trainloader = DataLoader(trainset, batch_size=bs, shuffle=shuffle)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    norm(torch.nn.Linear(3072, WIDTH), kind="one-inf", max_norm=MAX_NORM),
    # GroupSort(WIDTH//2),
    torch.nn.ReLU(),
    norm(torch.nn.Linear(WIDTH, WIDTH), kind="inf", max_norm=MAX_NORM),
    # GroupSort(WIDTH//2),
    torch.nn.ReLU(),
    norm(torch.nn.Linear(WIDTH, 100), kind="inf", max_norm=MAX_NORM),
).to(device)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# save initial model entirely
if WANDB:
    root = "/data/kitouni/LipNN-Bench/"
    os.makedirs(root + "checkpoints", exist_ok=True)

if OPTIM.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
elif OPTIM.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
else:
    raise ValueError("Optimizer {OPTIM} not supported")
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, total_steps=EPOCHS*len(trainloader))

pbar = tqdm(range(EPOCHS))
# make checkpoints folder
best = 0
x, y = next(iter(trainloader))
x /= x.max()
x, y = x.to(device), y.to(device)
for epoch in pbar:
    agg_loss = 0
    agg_acc = 0
    # for x, y in trainloader:
        # x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    pred = model(x)
    loss = torch.nn.functional.cross_entropy(TAU * pred, y)
    loss.backward()
    optimizer.step()
    # scheduler.step()
    with torch.no_grad(): acc = accuracy(pred, y)
    pbar.set_description(f"loss: {loss.item():.4f}, acc: {acc.item():.4f}")
    if WANDB: wandb.log({"loss": loss, "acc": acc})
    agg_loss += loss.item()
    agg_acc += acc.item()

    agg_loss /= len(trainloader)
    agg_acc /= len(trainloader)
    pbar.set_postfix_str(f"epoch_loss: {agg_loss:.4f}, epoch_acc: {agg_acc:.4f}")
    if WANDB:
        if TRACK_NORM: wandb.log(track_norms(model))
        wandb.log({"epoch_loss": agg_loss, "epoch_acc": agg_acc})
        if epoch % (EPOCHS//20) == 0:
            best = max(best, agg_acc)
            torch.save(model.state_dict(),
                       root + f"checkpoints/{epoch}_{best:.3f}.pt")
            wandb.save(root+ f"checkpoints/{epoch}_{best:.3f}.pt", base_path=root)


        
# %%
