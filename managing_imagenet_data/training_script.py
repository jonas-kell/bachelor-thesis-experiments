import os
import datetime
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CustomDataset import CustomDataset
from custom_imagenet_constants import (
    path_to_target_folder_for_transformed_data,
    tensorboard_log_folder,
)

run_name = "test_at_" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
writer = SummaryWriter(os.path.join(tensorboard_log_folder, run_name), flush_secs=1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(224 * 224 * 3, 512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 100),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# ! def loops


def train_loop(epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_correct = 0, 0
    loading_time, processing_time = 0, 0
    log_frequency = 50

    start_loading_time = time.time()
    for batch, (X, y) in enumerate(dataloader):
        # move to device
        X = X.to(device)
        y = y.to(device)

        # timing
        start_processing_time = end_loading_time = time.time()
        loading_time += end_loading_time - start_loading_time

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # aggregate log data
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item() / len(X)

        # timing
        end_processing_time = time.time()
        processing_time += end_processing_time - start_processing_time

        # log info to tensorboard and console
        if batch % log_frequency == 0 and batch != 0:
            # average data
            train_loss /= log_frequency
            train_correct /= log_frequency / 100.0
            loading_time /= log_frequency / 1000.0
            processing_time /= log_frequency / 1000.0

            # tensorboard logs
            writer.add_scalar("Loss/train", train_loss, epoch * num_batches + batch)
            writer.add_scalar(
                "Accuracy(%)/train", train_correct, epoch * num_batches + batch
            )
            writer.add_scalar(
                "Time(ms)/load", loading_time, epoch * num_batches + batch
            )
            writer.add_scalar(
                "Time(ms)/process", processing_time, epoch * num_batches + batch
            )

            # console logs
            print(
                f"Accuracy: {(train_correct):>0.1f}%, loss: {train_loss:>7f}  [{batch * len(X):>6d}/{size:>6d}]   load: {loading_time:>4f}ms, process: {processing_time:>4f}ms"
            )

            # reset counters
            train_loss, train_correct, loading_time, processing_time = 0, 0, 0, 0

        # timing
        start_loading_time = time.time()


def val_loop(epoch, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # move to device
            X = X.to(device)
            y = y.to(device)

            # test
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # average data
    val_loss /= num_batches
    val_correct /= size / 100.0

    # tensorboard logs
    writer.add_scalar("Accuracy(%)/test", val_correct, epoch)
    writer.add_scalar("Loss/test", val_loss, epoch)

    # console logs
    print(
        f"Validation Phase: \n Accuracy: {(val_correct):>0.1f}%, Avg loss: {val_loss:>8f} \n"
    )


if __name__ == "__main__":
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # model
    model = NeuralNetwork().to(device)

    # hyperparameters
    learning_rate = 1e-3
    epochs = 80  # epoch: train loop + validation/test
    batch_size = 128

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # datasets
    training_data = CustomDataset(path_to_target_folder_for_transformed_data, "train")
    val_data = CustomDataset(path_to_target_folder_for_transformed_data, "val")

    # dataloaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    # train it like it's hot
    for t in range(epochs):
        # run an epoch
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(t, train_dataloader, model, loss_fn, optimizer)
        val_loop(t, val_dataloader, model, loss_fn)

        # store the model
        print("Saving: ")
        torch.save(
            model,
            os.path.join(tensorboard_log_folder, run_name, "model_" + str(t) + ".pth"),
        )
    print("Done!")

    # close the writer
    writer.close()
