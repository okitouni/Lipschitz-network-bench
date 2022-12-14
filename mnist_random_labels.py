import torch
import tqdm
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from monotonenorm import direct_norm, GroupSort
from torchmetrics.functional import accuracy

# Set up the dataset
dataset = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Set up the model
max_norm = 2.0
lip_model = torch.nn.Sequential(
    torch.nn.Flatten(),
    direct_norm(torch.nn.Linear(784, 100), max_norm=max_norm, kind='one-inf'),
    GroupSort(100, 10),
    direct_norm(torch.nn.Linear(100, 10), max_norm=max_norm, kind='inf'),
)

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
)

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
pbar = tqdm.tqdm(range(100))
for epoch in pbar:
    for batch in dataloader:
        x, y = batch
        y = torch.randint(0, 10, (y.shape[0],), dtype=torch.long)
        pred = model(x)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(pred.softmax(dim=1), y, task='multiclass', num_classes=10)
    pbar.set_description(f'Epoch {epoch}, loss {loss.item():.3f}, acc {acc:.3f}')


