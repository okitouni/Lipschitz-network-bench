Code used to (over)fit Lipschitz networks on MNIST, CIFAR10, CIFAR100 with real and random labels as well as a CIFAR100 with an additional "goodness" feature $x\in [0,1]$ to showcase the monotonicity aspect of the architecture. The synthetic monotonicity problem is currently implemented such that samples with values above a critical threshold in the goodness feature $x>x_{\rm crit}$ are labeled 0. An alternative implementation is to take label 0 with probability $x$ and keep the original label (or assign a random one) with probability $1-x$.
### Requirements
torch, torchvision, tqdm, and monotonenorm 
monotonenorm can be installed as:
``pip install monotonenorm``
or 
``conda install -c okitouni monotonenorm``