# %%
from monotonenorm import direct_norm, GroupSort
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_layer(
    norm_func=None, act_func=None, depth=None, width=None, kind="one", max_norm=1,
):
    if norm_func is None:
        norm_func = lambda x, **kwargs: x
    if act_func is None:
        act_func = torch.nn.ReLU()
    return (
        norm_func(
            torch.nn.Linear(width, width),
            kind=kind,
            max_norm=max_norm,
            always_norm=False,
        ),
        act_func,
    )

# %%

torch.manual_seed(0)
depth = 10
width = 128
max_norm = 1  # ** (1 / depth)
gs_network = torch.nn.Sequential(
    torch.nn.Flatten(),
    direct_norm(
        torch.nn.Linear(784, width),
        kind="one-inf",
        max_norm=max_norm,
        always_norm=False,
    ),
    GroupSort(width//2),
    *[
        get_layer(direct_norm, GroupSort(width//2), depth=i, width=width, kind="inf")[layer]
        for i in range(depth - 2)
        for layer in range(2)
    ],
    direct_norm(
        torch.nn.Linear(width, 10),
        kind="inf",
        max_norm=max_norm,
        always_norm=False,
    ),
).to(device)
print(" parameters : ", sum(p.numel() for p in gs_network.parameters() if p.requires_grad))
# %%
gs_network = torch.nn.Sequential(
  torch.nn.Flatten(),
  torch.nn.Linear(784, width),
  torch.nn.ReLU(),
  *[
      get_layer(depth=i, width=width)[layer]
      for i in range(depth - 2)
      for layer in range(2)
  ],
  torch.nn.Linear(width, 10),
).to(device)
print(" parameters : ", sum(p.numel() for p in gs_network.parameters() if p.requires_grad))

# %%
import torchvision
import torchvision.transforms as transforms
import torch

# load MNIST
transform = transforms.Compose(
    [transforms.ToTensor()]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainset.targets = torch.randint(0, 10, (len(trainset),))

#move to device
data = trainset.data.to(device).float()
targets = trainset.targets.to(device)

# %%
print(gs_network)
# %%
optim = torch.optim.Adam(gs_network.parameters(), lr=5e-4)
loss_func = torch.nn.MultiMarginLoss(margin=.05)
#loss_func = torch.nn.CrossEntropyLoss()
bar = tqdm(range(100000))

# train loop

for epoch in bar:
    # subsample
    idx = torch.randint(0, len(data), (4096,))
    data_ = data[idx]
    targets_ = targets[idx]
    optim.zero_grad()
    outputs = gs_network(data_)
    loss = loss_func(outputs, targets_)
    loss.backward()
    optim.step()
    bar.set_description(f"loss: {loss.item():.4f}, acc: {torch.mean((torch.argmax(outputs, dim=1) == targets_).float()).item():.4f}")

# %%
