import random
import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
    
 def training(traindata, model, optimizer, examplestats, batch_size = 64):
  model.train()

  # reshuffle the training data set
  random_arrange = random.sample(range(len(traindata.targets)), len(traindata.targets))

  for i, start in enumerate(range(0, len(traindata.targets), batch_size)):
    # Obtain the indices of the training data set from the reshuffling
    batch = random_arrange[start: (start + batch_size)]

    # Get batch inputs and targets, transform them appropriately
    train_update = []
    convert_tensor = transforms.ToTensor()
    for i in batch:
      train_update.append(convert_tensor(traindata.__getitem__(i)[0])*255)
    input = torch.stack(train_update)
    target = torch.LongTensor(np.array(traindata.targets)[batch].tolist())

    # Forward propagation, compute loss, get predictions
    model.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)

    # Update statistics and loss
    acc = predicted == target
    train_indx = np.array(range(len(traindata.targets)))

    for j, index in enumerate(batch):
      original_index = train_indx[index]
      index_stats = examplestats.get(original_index, [[], [], []])
      index_stats[0].append(loss[j].item())
      index_stats[1].append(acc[j].sum().item())
      #index_stats[2].append(margin)
      examplestats[original_index] = index_stats

  loss = loss.mean()
  loss.backward()
  optimizer.step()
  #return examplestats
  
def forgetting(examplestats, epoch = 25):
  forgetness = {}
  firstlearn = {}
  for index, lst in examplestats.items():
    acc_full = np.array(lst[1])
    transition = acc_full[1:] - acc_full[:-1]
    if len(np.where(transition == -1)[0]) > 0:
      forgetness[index] = len(np.where(transition == -1)[0])
    elif len(np.where(acc_full == 1)[0]) == 0:
      forgetness[index] = epoch
    else:
      forgetness[index] = 0
    
  return dict(sorted(forgetness.items(), key = lambda item: item[1], reverse = True))

train = datasets.MNIST(root = "./data/", download = True, train = True)
CNN_Model = CNN().to('cpu')
CNN_Optim = optim.SGD(CNN_Model.parameters(), lr = 0.01, momentum = 0.5)
# Setup loss
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce=False)

example_stats = {}
epoch = 100
for i in range(epoch):
  training(train, CNN_Model, CNN_Optim, example_stats)
print(forgetting(example_stats, epoch = epoch))

### Testing - examples that never learnt, assume epoch = 100 (proportion of no learning about 2.7%)
c = forgetting(example_stats)
result = 0
for i,j in c.items():
  if j == 25:
    result += 1
print(result)
