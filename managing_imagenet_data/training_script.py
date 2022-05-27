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

run_date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
run_name = "test_at_" + run_date
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
    train_loss, train_correct, loading_time_hdd, loading_time_gpu, processing_time = (
        0,
        0,
        0,
        0,
        0,
    )
    log_frequency = 50

    start_loading_time_hdd = time.time()
    for batch, (X, y) in enumerate(dataloader):
        # timing
        start_loading_time_gpu = end_loading_time_hdd = time.time()
        loading_time_hdd += end_loading_time_hdd - start_loading_time_hdd

        # move to device
        X = X.to(device)
        y = y.to(device)

        # timing
        start_processing_time = end_loading_time_gpu = time.time()
        loading_time_gpu += end_loading_time_gpu - start_loading_time_gpu

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
            loading_time_hdd /= log_frequency / 1000.0
            loading_time_gpu /= log_frequency / 1000.0
            processing_time /= log_frequency / 1000.0

            # tensorboard logs
            writer.add_scalar("Loss/train", train_loss, epoch * num_batches + batch)
            writer.add_scalar(
                "Accuracy(%)/train", train_correct, epoch * num_batches + batch
            )
            writer.add_scalar(
                "Time(ms)/load_hdd", loading_time_hdd, epoch * num_batches + batch
            )
            writer.add_scalar(
                "Time(ms)/load_gpu", loading_time_gpu, epoch * num_batches + batch
            )
            writer.add_scalar(
                "Time(ms)/process", processing_time, epoch * num_batches + batch
            )

            # console logs
            print(
                f"Accuracy: {(train_correct):>0.1f}%, loss: {train_loss:>7f}  [{batch * len(X):>6d}/{size:>6d}]   load_hdd: {loading_time_hdd:>4f}ms, load_gpu: {loading_time_gpu:>4f}ms, process: {processing_time:>4f}ms"
            )

            # reset counters
            (
                train_loss,
                train_correct,
                loading_time_hdd,
                loading_time_gpu,
                processing_time,
            ) = (0, 0, 0, 0, 0)

        # timing
        start_loading_time_hdd = time.time()


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
    momentum = 0
    dampening = 0
    weight_decay = 0
    epochs = 80  # epoch: train loop + validation/test
    batch_size = 128

    loss_fn_name = "cross_entropy_loss"
    if loss_fn_name == "cross_entropy_loss":
        loss_fn = nn.CrossEntropyLoss()

    optimizer_name = "sgd"
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )

    # datasets
    training_data = CustomDataset(path_to_target_folder_for_transformed_data, "train")
    val_data = CustomDataset(path_to_target_folder_for_transformed_data, "val")

    # dataloaders
    nr_workers = 0
    pin_memory = (
        device == "cuda" and False
    )  # disabled, as this doesn't seem to help here
    shuffle = True
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nr_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=nr_workers,
        pin_memory=pin_memory,
    )

    # log hyperparameters to file
    writer.add_hparams(
        {
            "max_epochs": epochs,
            "batch_size": batch_size,
            "loss_fn": loss_fn_name,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nr_workers": nr_workers,
            "pin_memory": pin_memory,
            "shuffle": shuffle,
        },
        {
            "placeholder": 0
        },  # the hyperparameter module is used normally to compare used hyperparamaters for multiple runs in one "script-execution". I can only get information about the interesting metrics (loss/accuracy/...) after my model has trained sufficiently long. It may crash or be aborted earlier however, what would result in not writing the corresponding hyperparameter entry. As the evaluating is done manually anyway, just a placeholder metric is inserted, to allow for the logging of hyperparameters for now. (no metric results in nothing being logged/displayed at all)
        run_name=run_date,
    )

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
