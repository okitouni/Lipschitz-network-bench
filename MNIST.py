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
width = 256
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
from torch.utils.data import DataLoader

# load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

trainloader = DataLoader(
    trainset, batch_size=2048, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=len(testset), shuffle=False, num_workers=2
)
# %%
print(gs_network)
# %%
optim = torch.optim.Adam(gs_network.parameters(), lr=1e-2)
loss_func = torch.nn.MultiMarginLoss(margin=.2)
#loss_func = torch.nn.CrossEntropyLoss()
bar = tqdm(range(200))

# train loop

for epoch in bar:
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outputs = gs_network(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optim.step()
        bar.set_description(f"loss: {loss.item():.4f}, acc: {torch.mean((torch.argmax(outputs, dim=1) == labels).float()).item():.4f}")

# %%
