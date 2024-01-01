import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from grid_load_single import GeoDataLoader, num_categories, width, height
import matplotlib.pyplot as plt
import numpy as np
import random
from avg_coords import *
from math import sqrt

batch_size = 1
num_images = 50000 # change to number of images used
test_data = GeoDataLoader("Training", int(num_images * 0.8))
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(width*height*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_categories)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()

def test(dataloader, model, loss_fn, dataset):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    progress = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            prediction = pred.max(1).indices
            correct += (prediction == y).type(torch.float).sum().item()
            progress += 1
            if progress % 1000 == 0:
                print(f'{progress}/{int(num_images * 0.8)}')
    test_loss /= num_batches
    correct /= size
    print(f"{dataset} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def try_model():
    index = random.randrange(0, len(test_data))
    x, y = test_data[index][0], test_data[index][1]
    img = np.transpose(x.detach().numpy(), axes=[1,2,0])
    imgplot = plt.imshow(img)
    x = torch.unsqueeze(x, 0)
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        index = pred[0].tolist().index(torch.max(pred[0]).item())
        #predicted, actual = states[index], states[y]
        #print(f'Predicted: "{predicted}", Actual: "{actual}"')
        #plt.title(f'Predicted: "{predicted}", Actual: "{actual}"')
        plt.show()

def distance(p0, p1):
    return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def get_avg_distance(dataloader, model, avg_coords):
    total_distance = 0
    progress = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            index = pred[0].tolist().index(torch.max(pred[0]).item())
            guess = avg_coords[index]
            answer = test_data.get_coords(progress)
            total_distance += distance(guess, answer)
            progress += 1
            if progress % 1000 == 0:
                print(f'{progress}/{int(num_images * 0.2)}')
    return total_distance / num_images

model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_categories)
model = model.to(device)
model.load_state_dict(torch.load("grid_model.pth"))
model.eval()

test(test_dataloader, model, loss_fn, "Train")
print(get_avg_distance(test_dataloader, model, grid_avg) * 69, "miles")
#try_model()
