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
    train_loss = torch.tensor([0], dtype=torch.float32, device=device)
    train_correct_top1 = torch.tensor([0], dtype=torch.float32, device=device)
    train_correct_top3 = torch.tensor([0], dtype=torch.float32, device=device)
    train_correct_top5 = torch.tensor([0], dtype=torch.float32, device=device)
    loading_time_hdd = 0
    loading_time_gpu = 0
    processing_time = 0

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
        train_loss += loss
        topk_pred = torch.topk(pred, 5, 1).indices.t()
        correct = topk_pred.eq(y.view(1, -1).expand_as(topk_pred))
        train_correct_top1 += correct[:1].float().sum() / X.shape[0]
        train_correct_top3 += correct[:3].float().sum() / X.shape[0]
        train_correct_top5 += correct[:5].float().sum() / X.shape[0]
        torch.cuda.synchronize()

        # timing
        end_processing_time = time.time()
        processing_time += end_processing_time - start_processing_time

        # log info to tensorboard and console
        if batch % log_frequency == 0 and batch != 0:
            # average data
            train_loss /= log_frequency
            train_correct_top1 /= log_frequency / 100.0
            train_correct_top3 /= log_frequency / 100.0
            train_correct_top5 /= log_frequency / 100.0
            loading_time_hdd /= log_frequency / 1000.0
            loading_time_gpu /= log_frequency / 1000.0
            processing_time /= log_frequency / 1000.0

            # tensorboard logs
            writer.add_scalar("Loss/train", train_loss, epoch * num_batches + batch)
            writer.add_scalar(
                "Accuracy(%)/train", train_correct_top1, epoch * num_batches + batch
            )
            writer.add_scalar(
                "Accuracy(%)/train_top3",
                train_correct_top3,
                epoch * num_batches + batch,
            )
            writer.add_scalar(
                "Accuracy(%)/train_top5",
                train_correct_top5,
                epoch * num_batches + batch,
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
                f"Acc: {(train_correct_top1.item()):>0.1f}%, Acc_t3: {(train_correct_top3.item()):>0.1f}%, Acc_t5: {(train_correct_top5.item()):>0.1f}%, loss: {train_loss.item():>7f}  [{batch * len(X):>6d}/{size:>6d}]   load_hdd: {loading_time_hdd:>4f}ms, load_gpu: {loading_time_gpu:>4f}ms, process: {processing_time:>4f}ms"
            )

            # reset counters
            train_loss = torch.tensor([0], dtype=torch.float32, device=device)
            train_correct_top1 = torch.tensor([0], dtype=torch.float32, device=device)
            train_correct_top3 = torch.tensor([0], dtype=torch.float32, device=device)
            train_correct_top5 = torch.tensor([0], dtype=torch.float32, device=device)
            loading_time_hdd = 0
            loading_time_gpu = 0
            processing_time = 0

        # timing
        start_loading_time_hdd = time.time()


def val_loop(epoch, dataloader, model, loss_fn, device, writer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss = torch.tensor([0], dtype=torch.float32, device=device)
    val_correct_top1 = torch.tensor([0], dtype=torch.float32, device=device)
    val_correct_top3 = torch.tensor([0], dtype=torch.float32, device=device)
    val_correct_top5 = torch.tensor([0], dtype=torch.float32, device=device)

    with torch.no_grad():
        for X, y in dataloader:
            # move to device
            X = X.to(device)
            y = y.to(device)

            # test
            pred = model(X)

            # log aggregation
            val_loss += loss_fn(pred, y)
            topk_pred = torch.topk(pred, 5, 1).indices.t()
            correct = topk_pred.eq(y.view(1, -1).expand_as(topk_pred))
            val_correct_top1 += correct[:1].float().sum()
            val_correct_top3 += correct[:3].float().sum()
            val_correct_top5 += correct[:5].float().sum()

    # average data
    val_loss /= num_batches
    val_correct_top1 /= size / 100.0
    val_correct_top3 /= size / 100.0
    val_correct_top5 /= size / 100.0

    # tensorboard logs
    writer.add_scalar("Accuracy(%)/test", val_correct_top1, epoch)
    writer.add_scalar("Accuracy(%)/test_top3", val_correct_top3, epoch)
    writer.add_scalar("Accuracy(%)/test_top5", val_correct_top5, epoch)
    writer.add_scalar("Loss/test", val_loss, epoch)

    # console logs
    print(
        f"Validation Phase: \n Acc: {(val_correct_top1.item()):>0.1f}%, Acc_t3: {(val_correct_top3.item()):>0.1f}%, Acc_t5: {(val_correct_top5.item()):>0.1f}%, Avg loss: {val_loss.item():>8f} \n"
    )


def train_model(
    device: Literal["cuda", "cpu"],
    network: nn.Module,
    constants: PathAndFolderConstants,
    mapper: SynsetMapper,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    dampening: float = 0,
    weight_decay: float = 0,
    max_epochs: int = 80,  # epoch: train loop + validation/test
    batch_size: int = 128,
    loss_fn_name: Literal["cross_entropy_loss"] = "cross_entropy_loss",
    optimizer_name: Literal["sgd", "adamw"] = "sgd",
    model_name: str = "",  # for log purposes only !!!
    preload_data_to_ram: bool = False,
    start_epoch: int = 0,
    is_continuing_training: bool = False,
    continue_training_path: str = "",
):
    # model to device
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
    if optimizer_name == "adamw":
        if weight_decay == 0:
            weight_decay = 1e-2
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    # restore model if needed
    if is_continuing_training and start_epoch > 0:
        stored_path = os.path.join(
            continue_training_path, "model_" + str(start_epoch - 1) + ".pth"
        )
        print(f"Restoring model from file '{stored_path}'")
        checkpoint = torch.load(stored_path)
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # datasets
    training_data = CustomDataset(
        constants, mapper, constants.train_folder_name, preload_data_to_ram
    )
    val_data = CustomDataset(
        constants, mapper, constants.val_folder_name, preload_data_to_ram
    )

    # dataloaders
    nr_workers = 4
    pin_memory = (
        device == "cuda"
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

    tensorboard_folder = "UNSET"
    # log writer
    flush_secs = 2
    if is_continuing_training:
        tensorboard_folder = continue_training_path
    else:
        run_date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_name = model_name + "_at_" + run_date

        tensorboard_folder = os.path.join(
            constants.path_to_tensorboard_log_folder, run_name
        )

    writer = SummaryWriter(
        tensorboard_folder,
        flush_secs=flush_secs,
    )

    if not is_continuing_training:
        # log hyperparameters to file
        writer.add_hparams(
            {
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "loss_fn_name": loss_fn_name,
                "optimizer_name": optimizer_name,
                "learning_rate": learning_rate,
                "momentum": momentum,
                "dampening": dampening,
                "weight_decay": weight_decay,
                "nr_workers": nr_workers,
                "pin_memory": pin_memory,
                "shuffle": shuffle,
                "model_name": model_name,
                "preload_data_to_ram": preload_data_to_ram,
            },
            {
                "placeholder": 0
            },  # the hyperparameter module is used normally to compare used hyperparamaters for multiple runs in one "script-execution". I can only get information about the interesting metrics (loss/accuracy/...) after my model has trained sufficiently long. It may crash or be aborted earlier however, what would result in not writing the corresponding hyperparameter entry. As the evaluating is done manually anyway, just a placeholder metric is inserted, to allow for the logging of hyperparameters for now. (no metric results in nothing being logged/displayed at all)
            run_name="./",
        )

    # slack
    if not is_continuing_training:
        post_message_to_slack("Training started")
    else:
        post_message_to_slack("Training continued")

    # train it like it's hot
    for t in range(start_epoch, max_epochs):
        # run an epoch
        epoch_message = f"Epoch {t}"
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
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(
                    tensorboard_folder,
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
