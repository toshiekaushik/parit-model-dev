from .model import TaskSchedulerPersonalizationV2
import torch
import torch.nn as nn


def train_on_dummy_data():
    torch.manual_seed(42)

    descriptors = torch.rand(100, 5)
    target = torch.rand(100, 1)

    model = TaskSchedulerPersonalizationV2(input_dim=5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epochs = 100
    for epoch in range(epochs):
        pred = model.foward(descriptors)
        loss = criterion(pred, target)

        optimizer.zero_grad()  # clear gradient
        loss.backward()  # back propogation
        optimizer.step()  # update wrights

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    weight, bias = model.parameters()
    print(f"\nLearned Weights: {weight} ")
    print(f"Learned Bias: {bias}")

train_on_dummy_data()
