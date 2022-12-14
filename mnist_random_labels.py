# %%
import torch
import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from monotonenorm import direct_norm, GroupSort
from torchmetrics.functional import accuracy
from gradient_descent_the_ultimate_optimizer import gdtuo
import wandb

wandb.init(project="MNIST-Rand", entity="iaifi", name="MNIST-Rand-MLP-GDUOAdam-CE-100-100-5-1e-3")

# Set up the dataset
dataset = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
# randomize labels
dataset.targets = torch.randint(0, 10, (len(dataset),))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
# %%
print(dataset[0][0].shape)
# %%
# Set up the model
# max_norm = 2.0
# lip_model = torch.nn.Sequential(
#     torch.nn.Flatten(),
#     direct_norm(torch.nn.Linear(784, 100), max_norm=max_norm, kind='one-inf'),
#     GroupSort(100, 10),
#     direct_norm(torch.nn.Linear(100, 10), max_norm=max_norm, kind='inf'),
# )

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
)

LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 128

wandb.config = {
    "learning_rate": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "model": "MLP",
    "optimizer": "GDUO-Adam",
    "loss": "CrossEntropy",
    "dataset": "CIFAR10",
    "width": 100,
    "depth": 5,
    "activation": "ReLU",
}

# Set up the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
mw = gdtuo.ModuleWrapper(model, optimizer=optimizer)
mw.initialize()
# Train the model
pbar = tqdm.tqdm(range(100))
for epoch in pbar:
    agg_acc = 0
    agg_loss = 0
    for batch in dataloader:
        x, y = batch
        mw.begin() # call this before each step, enables gradient tracking on desired params
        pred = mw.forward(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        mw.zero_grad()
        loss.backward(create_graph=True) # important! use create_graph=True
        mw.step()
        # x, y = batch
        # pred = model(x)
        # optimizer.zero_grad()
        # loss = torch.nn.functional.cross_entropy(pred, y)
        # loss.backward()
        # optimizer.step()
        acc = accuracy(pred, y, num_classes=10)
        pbar.set_description(f'Epoch {epoch}, loss {loss.item():.3f}, acc {acc:.3f}')
        agg_acc += acc/len(dataloader)
        agg_loss += loss.item()/len(dataloader)
    wandb.log({"loss": agg_loss, "acc": agg_acc})
    wandb.watch(model)

