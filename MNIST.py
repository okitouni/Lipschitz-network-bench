# %%
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from monotonenorm import direct_norm, GroupSort
import torch
from tqdm import tqdm
import os

torch.manual_seed(1)

BATCHSIZE = -1
EPOCHS = 200000
RANDOM_LABELS = True
MODEL = "Lipschitz"  # "Lipschitz" or "Unconstrained"
TAU = 256  # rescale temperature of CrossEntropyLoss
MAX_NORM = 1  # max norm of each layer
DATASET = "MNIST"
WIDTH = 1024
LR = 1e-5
OPTIM = "Adam"
TRACK_NORM = False
WANDB = False

name = f"{DATASET}_{MODEL}_{WIDTH}_tau{TAU}_maxnorm{MAX_NORM}"
if RANDOM_LABELS:
    name += "_random"
if WANDB:
    import wandb
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
        "activation": "GroupSort",
    }


norm = direct_norm if MODEL == "Lipschitz" else lambda x, **kwargs: x
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

if RANDOM_LABELS:
    trainset.targets = torch.randint(0, 10, (len(trainset),))
bs = BATCHSIZE if BATCHSIZE > 0 else len(trainset)
shuffle = True if BATCHSIZE > 0 else False

trainloader = DataLoader(trainset, batch_size=bs, shuffle=shuffle)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    norm(torch.nn.Linear(784, WIDTH), kind="one-inf", max_norm=MAX_NORM),
    GroupSort(WIDTH // 2),
    # torch.nn.ReLU(),
    norm(torch.nn.Linear(WIDTH, WIDTH), kind="inf", max_norm=MAX_NORM),
    GroupSort(WIDTH // 2),
    # torch.nn.ReLU(),
    norm(torch.nn.Linear(WIDTH, 10), kind="inf", max_norm=MAX_NORM),
).to(device)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# save initial model entirely
if WANDB:
    root = "/data/kitouni/LipNN-Bench/"
    if not os.path.exists(root):
        raise ValueError("{root} does not exist please update root variable")
    os.makedirs(root + "checkpoints", exist_ok=True)
    wandb.save(f"{__file__}")

if OPTIM.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
elif OPTIM.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
else:
    raise ValueError("Optimizer {OPTIM} not supported")
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, total_steps=EPOCHS*len(trainloader))

pbar = tqdm(range(EPOCHS))
# make checkpoints folder
x, y = next(iter(trainloader))
x /= x.max()
x, y = x.to(device), y.to(device)
for epoch in pbar:
    # for x, y in trainloader:
    # x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    pred = model(x)
    loss = torch.nn.functional.cross_entropy(TAU * pred, y)
    loss.backward()
    optimizer.step()
    # scheduler.step()
    with torch.no_grad():
        acc = (pred.argmax(dim=1) == y).float().mean()
    pbar.set_description(f"loss: {loss.item():.4f}, acc: {acc.item():.4f}")
    if WANDB:
        wandb.log({"loss": loss, "acc": acc})
        if epoch % (EPOCHS // 20) == 0:
            torch.save(model.state_dict(), root + f"checkpoints/{DATASET}_{epoch}.pt")
            wandb.save(root + f"checkpoints/{DATASET}_{epoch}.pt", base_path=root)


# %%
