import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from model import ModelAI
from dataset import MyDataset
from loss import LossFunction

seed = 100
torch.manual_seed(seed)

LEARNING_RATE = 0.001
DEVICE = "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHES = 100
NUM_WORKS = 0
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth.tar"

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm.tqdm(train_loader, leave = True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        # x, y là dữ liệu đầu vào của ta x = text_matrix, y = [prediction]
        # vd: x = torch.randn(2, 1, 13, 13)
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        print(f"loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = loss.item())
        
def main():
    model = ModelAI(in_channels=1, dataSize = 4, layers = 128)
    optimizer = optim.Adam(
        model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = LossFunction()
    if LOAD_MODEL:
        pass
    
    train_dataset = MyDataset(
        "dataset_train.txt",
        out_size=1
    )
    test_dataset = MyDataset(
        "dataset_test.txt",
        out_size=1
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    for epoch in range(EPOCHES):
        train_fn(train_loader, model, optimizer, loss_fn)
main()