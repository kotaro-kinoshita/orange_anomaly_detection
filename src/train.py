import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import TrainDataset
from model.auto_encorder import AutoEncoder
from loss import SSIM_Loss

from PIL import Image

PATH_DATASET = "../../data/dataset/normal"
LOG = "../logs/AE_Checkpoint_MSE"
BATCH_SIZE = 64
EPOCH = 50000
CHECK_POINT = 10000
WINDOW_SIZE = 160

os.makedirs(LOG, exist_ok=True)

dataset = TrainDataset(PATH_DATASET, size=WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#criterion = SSIM_Loss(data_range=1, size_average=True, channel=3)
criterion = nn.MSELoss(reduction="sum")

model = AutoEncoder(d=128).cuda()
optimizer =  optim.Adam(model.parameters(), lr=2*1e-4)

for epoch in range(EPOCH):
    for iter, data in enumerate(dataloader):
        img, _ = data
        input = Variable(img).cuda()
        output = model(input)
        loss = criterion(output, input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch:{epoch}/{EPOCH}, iter:{iter}, loss:{loss.item():.4f}")

    if (epoch + 1) % CHECK_POINT == 0:
        torch.save(model.state_dict(), f"{LOG}/conv_autoencoder_{epoch+1}.pth")







