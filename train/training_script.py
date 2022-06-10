import os
import datetime
import time
import traceback
from typing import Literal


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from CustomDataset import CustomDataset
from PathAndFolderConstants import PathAndFolderConstants
from SynsetMapper import SynsetMapper
from message import post_message_to_slack


def train_loop(epoch, dataloader, model, loss_fn, optimizer, device, writer):
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


def val_loop(epoch, dataloader, model, loss_fn, device, writer):
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


def train_model(
    device: Literal["cuda", "cpu"],
    network: nn.Module,
    constants: PathAndFolderConstants,
    mapper: SynsetMapper,
    learning_rate: float = 1e-3,
    momentum: float = 0,
    dampening: float = 0,
    weight_decay: float = 0,
    epochs: int = 80,  # epoch: train loop + validation/test
    batch_size: int = 128,
    loss_fn_name: Literal["cross_entropy_loss"] = "cross_entropy_loss",
    optimizer_name: Literal["sgd"] = "sgd",
):
    # model
    model = network.to(device)

    if loss_fn_name == "cross_entropy_loss":
        loss_fn = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )

    # datasets
    training_data = CustomDataset(constants, mapper, constants.train_folder_name)
    val_data = CustomDataset(constants, mapper, constants.val_folder_name)

    # dataloaders
    nr_workers = 4
    pin_memory = (
        device == "cuda" and True
    )  # faster and larger memory and easy peasy everything is faster and pinning works
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

    # log writer
    run_date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    run_name = "test_at_" + run_date
    writer = SummaryWriter(
        os.path.join(constants.path_to_tensorboard_log_folder, run_name), flush_secs=1
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
        run_name="./",
    )

    # slack
    post_message_to_slack("Training started")

    # train it like it's hot
    for t in range(epochs):
        # run an epoch
        epoch_message = f"Epoch {t+1}"
        print(epoch_message + "\n-------------------------------")
        writer.add_text("epoch", epoch_message)

        try:
            train_loop(t, train_dataloader, model, loss_fn, optimizer, device, writer)
        except Exception as exc:
            writer.add_text("error_train", traceback.format_exc())
            post_message_to_slack("Error: error_train")
            raise exc

        try:
            val_loop(t, val_dataloader, model, loss_fn, device, writer)
        except Exception as exc:
            writer.add_text("error_val", traceback.format_exc())
            post_message_to_slack("Error: error_val")
            raise exc

        # store the model
        print("Saving: ")
        try:
            torch.save(
                model,
                os.path.join(
                    constants.path_to_tensorboard_log_folder,
                    run_name,
                    "model_" + str(t) + ".pth",
                ),
            )
        except Exception as exc:
            writer.add_text("error_save", traceback.format_exc())
            post_message_to_slack("Error: error_save")
            raise exc

    print("Done!")
    # slack
    post_message_to_slack("Training completed")

    # close the writer
    writer.close()
