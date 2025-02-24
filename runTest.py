import torch
import torch.nn as nn
import model
import loss
import dataset
md = model.ModelAI(in_channels=1, dataSize = 4, layers = 128)
data = dataset.MyDataset("dataset_train.txt", 1)
a = data.getitem(0)[0].unsqueeze(0)
print(md(a).shape)