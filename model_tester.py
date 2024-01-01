import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from state_load_single import GeoDataLoader, num_categories, width, height, states
# can change above file to state, grid, or cluster models

batch_size = 32
num_images = 1000 # change to number of images used
training_data = GeoDataLoader("Training", int(num_images * 0.8))
test_data = GeoDataLoader("Testing", int(num_images * 0.2))

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
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

class ConvNeuralNetwork(nn.Module):
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

#model = NeuralNetwork().to(device)

#"""
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_categories)
model = model.to(device)
#"""
#breakpoint()

print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, dataset, progress):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            prediction = pred.max(1).indices
            correct += (prediction == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{dataset} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    progress.append((100*correct, test_loss))

epochs = 1
test_progress = []
train_progress = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, "Test", test_progress)
    #test(train_dataloader, model, loss_fn, "Train", train_progress)
print("Done!")

torch.save(model.state_dict(), "test_model.pth")
print("Saved PyTorch Model State to model.pth")
print()
print(test_progress)
print(train_progress)

model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_categories)
model = model.to(device)
model.load_state_dict(torch.load("test_model.pth"))
model.eval()

test(test_dataloader, model, loss_fn, "Test", test_progress)
x, y = test_data[0][0], test_data[0][1]
x = torch.unsqueeze(x, 0)
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    index = pred[0].tolist().index(torch.max(pred[0]).item())
    predicted, actual = states[index], states[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
