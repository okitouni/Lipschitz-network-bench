# %%
import torch
import tqdm
from torchvision.datasets import CIFAR100
from monotonenorm import direct_norm, GroupSort
from torchmetrics.functional import accuracy

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

LR = 5e-2
EPOCHS = 10000
N_TARGETS = 100

# Set up the dataset
dataset = CIFAR100(root='data', train=True, download=True)
X = torch.tensor(dataset.data).to(device).float() / 255
#random labels
Y = torch.randint(0, N_TARGETS, (len(dataset),)).to(device)
# %%
# Set up the model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(32*32*3, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, N_TARGETS),
).to(device)
print(" parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# %%

# Set up the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
#lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=1, epochs=EPOCHS, anneal_strategy='linear', final_div_factor=1e3)
# optimizer = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
# mw = gdtuo.ModuleWrapper(model, optimizer=optimizer)
# mw.initialize()
# Train the model
pbar = tqdm.tqdm(range(EPOCHS))
for epoch in pbar:
    agg_loss = 0
    agg_acc = 0
    # mw.begin() # call this before each step, enables gradient tracking on desired params
    # pred = mw.forward(x)
    # loss = torch.nn.functional.cross_entropy(pred, y)
    # mw.zero_grad()
    # loss.backward(create_graph=True) # important! use create_graph=True
    # mw.step()

    pred = model(X)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(pred, Y)
    loss.backward()
    optimizer.step()
    #lr_scheduler.step()

    acc = accuracy(pred, Y, task="multiclass", num_classes=N_TARGETS)
    pbar.set_description(f'Epoch {epoch}, loss {loss.item():.3f}, acc {acc:.3f}, lr {optimizer.param_groups[0]["lr"]:.3e}')



# %%
