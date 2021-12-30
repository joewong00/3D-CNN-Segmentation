from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms
import matplotlib.pyplot as plt
from residual3dunet.model import ResidualUNet3D
from tqdm import tqdm
import torch
from utils import get_loaders, check_accuracy, load_checkpoint, save_checkpoint

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240 
IMAGE_DEPTH = 14
PIN_MEMORY = True
LOAD_MODEL = False


def train_net(loader, model, optimizer, loss, scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    model = ResidualUNet3D(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, train=True)
    val_loader = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, train=False)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_net(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()