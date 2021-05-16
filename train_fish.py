import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torchvision.io import read_image
import torch.optim as optim
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

REBUILD_DATA = False # Set to true to re process training data

class Fish(): # Import and preprocess data set 

  img_size = 50
  black_sea_sprat = "Fish_Dataset/Fish_Dataset/Black Sea Sprat/Black Sea Sprat"
  gilt_head_bream = "Fish_Dataset/Fish_Dataset/Gilt-Head Bream/Gilt-Head Bream"
  horse_mackerel = "Fish_Dataset/Fish_Dataset/Hourse Mackerel/Hourse Mackerel"
  red_mullet = "Fish_Dataset/Fish_Dataset/Red Mullet/Red Mullet"
  red_sea_bream = "Fish_Dataset/Fish_Dataset/Red Sea Bream/Red Sea Bream"
  sea_bass = "Fish_Dataset/Fish_Dataset/Sea Bass/Sea Bass"
  shrimp = "Fish_Dataset/Fish_Dataset/Shrimp/Shrimp"
  striped_red_mullet = "Fish_Dataset/Fish_Dataset/Striped Red Mullet/Striped Red Mullet"
  trout = "Fish_Dataset/Fish_Dataset/Trout/Trout"

  LABELS = {black_sea_sprat : 0, gilt_head_bream : 1, horse_mackerel : 2, red_mullet : 3, red_sea_bream : 4, sea_bass : 5, shrimp : 6, striped_red_mullet : 7, trout : 8}

  training_data = []

  black_sea_sprat_count = 0
  gilt_head_bream_count = 0
  horse_mackerel_count = 0 
  red_mullet_count = 0 
  red_sea_bream_count = 0 
  sea_bass_count = 0 
  shrimp_count = 0 
  striped_red_mullet_count = 0 
  trout_count = 0 
   
  def make_training_data(self):
    for label in self.LABELS:
      print(label)
      for f in tqdm(os.listdir(label)):
        try:
          path = os.path.join(label, f)
          img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
          img = cv2.resize(img, (self.img_size, self.img_size))
          self.training_data.append([np.array(img), np.eye(9)[self.LABELS[label]]])

          # Count the number of images for each class of fish
          if label == self.black_sea_sprat:
            self.black_sea_sprat_count += 1
          elif label == self.gilt_head_bream:
            self.gilt_head_bream_count += 1
          elif label == self.horse_mackerel:
            self.horse_mackerel_count += 1
          elif label == self.red_mullet:
            self.red_mullet_count += 1
          elif label == self.red_sea_bream:
            self.red_sea_bream_count += 1        
          elif label == self.sea_bass:
            self.sea_bass_count += 1
          elif label == self.shrimp:
            self.shrimp_count += 1
          elif label == self.striped_red_mullet:
            self.striped_red_mullet_count += 1
          elif label == self.trout:
            self.trout_count += 1

        except Exception as e:
          pass

    np.random.shuffle(self.training_data)
    np.save("training_data.npy", self.training_data)
    
    # Print the number of images in data set for each class of fish
    print("black_sea_sprat :", self.black_sea_sprat_count)
    print("gilt_head_bream :", self.gilt_head_bream_count)
    print("horse_mackerel :", self.horse_mackerel_count)
    print("red_mullet :", self.red_mullet_count)
    print("red_sea_bream :", self.red_sea_bream_count) 
    print("sea_bass :", self.sea_bass_count)
    print("shrimp :", self.shrimp_count)
    print("striped_red_mullet :", self.striped_red_mullet_count)
    print("trout :", self.trout_count)

class Net(nn.Module): # CNN model
  def __init__(self):
    super().__init__()
    
    # 2D convolution is used on 2D data set like images. 3D would be useful for 3D models and scans.
    self.conv1 = nn.Conv2d(1, 32, 5) # Input, number of convolutional features, kernel size
    self.conv2 = nn.Conv2d(32, 64, 5) # Output from previous convolutional layer is input for following layer
    self.conv3 = nn.Conv2d(64, 128, 5)
    self.conv4 = nn.Conv2d(128, 256, 5)
    self.conv5 = nn.Conv2d(256, 512, 5)
    self.conv6 = nn.Conv2d(512, 1024, 5)

    x = torch.randn(50, 50).view(-1, 1, 50, 50)
    self._to_linear = None
    self.convs(x)

    self.fc1 = nn.Linear(self._to_linear, 512) # First Fully connected layer
    self.fc2 = nn.Linear(512, 9) # Second Fully connected layer for 9 classes
  
  def convs(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
    # x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
    # x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
    # x = F.max_pool2d(F.relu(self.conv6(x)), (2, 2))

    # print(x[0].shape)

    if self._to_linear is None:
      self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

    return x
  
  def forward(self, x):

    x = self.convs(x)
    x = x.view(-1, self._to_linear)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return F.softmax(x, dim=1) # Activation Function


if __name__ == "__main__":

  if REBUILD_DATA:
    fish = Fish()
    fish.make_training_data()

  training_data = np.load("training_data.npy", allow_pickle= True)
  net = Net()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # lr is learning rate
  loss_function = nn.MSELoss()

  X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) # Images
  X = X / 255.0
  y = torch.Tensor([i[1] for i in training_data]) # Image lables

  VAL_PCT = 0.8 # Percent of data to train with
  val_size = int(len(X) * VAL_PCT)
  # print(val_size)

  train_X = X[: -val_size]
  train_y = y[: -val_size]

  test_X = X[-val_size:]
  test_y = y[-val_size:]

  BATCH_SIZE = 100

  EPOCHS = 60

  for epoch in range(EPOCHS): # View the slicing of data for a given BATCH_SIZE
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # tqdm shows progress bar
      # print(i, i + BATCH_SIZE)
      batch_X = train_X[i : i + BATCH_SIZE].view(-1, 1, 50, 50)
      batch_y = train_y[i : i + BATCH_SIZE]

      # Zero the gradient 
      net.zero_grad()
      outputs = net(batch_X)
      loss = loss_function(outputs, batch_y)
      loss.backward()
      optimizer.step()
  print("Loss: ", loss)

  correct = 0
  total = 0

  with torch.no_grad():
    for i in tqdm(range(len(test_X))):
      real_class = torch.argmax(test_y[i])
      net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
      predicted_class = torch.argmax(net_out)
      if predicted_class == real_class:
        correct += 1
      
      total += 1

  accuracy = round(correct / total, 3)
  print("Accuracy: ", accuracy)

  plt.plot()