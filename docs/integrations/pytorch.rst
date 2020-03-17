PyTorch
=======

Log PyTorch metrics
-------------------
I have a training script written in `PyTorch <https://pytorch.org>`_. How do I adjust it to log metrics to Neptune?

.. image:: ../_static/images/others/pytorch_neptuneml.png
   :target: ../_static/images/others/pytorch_neptuneml.png
   :alt: Pytorch neptune.ai integration

Say your training script looks like this:

.. code-block::

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.optim as optim
   from torchvision import datasets, transforms

   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ITERATIONS = 10000

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(1, 20, 5, 1)
           self.conv2 = nn.Conv2d(20, 50, 5, 1)
           self.fc1 = nn.Linear(4*4*50, 500)
           self.fc2 = nn.Linear(500, 10)

       def forward(self, x):
           x = F.relu(self.conv1(x))
           x = F.max_pool2d(x, 2, 2)
           x = F.relu(self.conv2(x))
           x = F.max_pool2d(x, 2, 2)
           x = x.view(-1, 4*4*50)
           x = F.relu(self.fc1(x))
           x = self.fc2(x)
           return F.log_softmax(x, dim=1)

   train_loader = torch.utils.data.DataLoader(
       datasets.MNIST('../data',
                      train=True,
                      download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))])
                      ),
       batch_size=64,
       shuffle=True)

   model = Net().to(DEVICE)

   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.to(DEVICE), target.to(DEVICE)
       optimizer.zero_grad()
       output = model(data)
       loss = F.nll_loss(output, target)
       loss.backward()
       optimizer.step()

       if batch_idx == ITERATIONS:
           break

Add a snippet to the training loop, that sends your loss or metric to Neptune:

.. code-block::

   import neptune

   neptune.init('shared/onboarding')
   neptune.create_experiment()
   ...
   for batch_idx, (data, target) in enumerate(train_loader):
       ...
       neptune.send_metric('batch_loss', batch_idx, loss.data.cpu().numpy())

Your loss is now logged to Neptune:

.. image:: ../_static/images/how-to/ht-log-pytorch-1.png
   :target: ../_static/images/how-to/ht-log-pytorch-1.png
   :alt: image
