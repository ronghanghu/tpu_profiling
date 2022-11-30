import argparse
import datetime
import multiprocessing
import os
import time

import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp

import utils
import models


NUM_CLASSES = 1000


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--model", default="ResNet50", type=str)
    parser.add_argument("--data_path", default="/datasets02/imagenet-1k/", type=str)
    parser.add_argument("--use_fake_data", action="store_true")
    parser.add_argument("--image_size", default=224, type=int)

    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--adamw_b1", type=float, default=0.9)
    parser.add_argument("--adamw_b2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--log_every_steps", default=20, type=int)
    parser.add_argument("--checkpoint_interval", default=10, type=int)
    parser.add_argument("--eval_interval", default=10, type=int)

    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--half_precision", action="store_true")

    # parameters for speed tweaking
    parser.add_argument("--use_tanh_for_gelu", action="store_true")
    parser.add_argument("--use_attention_with_einsum", action="store_true")
    parser.add_argument("--no_pin_layout", action="store_false", dest="pin_layout")
    parser.set_defaults(pin_layout=True)
    return parser


def main(config):
    xm.master_print(f"config = {config}\n".replace(", ", ",\n"))

    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    device = xm.xla_device()
    assert config.batch_size % world_size == 0
    local_batch_size = config.batch_size // world_size
    os.makedirs(config.output_dir, exist_ok=True)

    # fix the seed for reproducibility
    seed = rank
    torch.manual_seed(seed)

    # load the training and validation data
    if config.use_fake_data:
        dataset_train = utils.FakeDataset(1281167, config.image_size, NUM_CLASSES)
        dataset_val = utils.FakeDataset(50000, config.image_size, NUM_CLASSES)
    else:
        # simple augmentation
        transform_train = T.Compose(
            [
                T.RandomResizedCrop(config.image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # simple center-crop
        transform_val = T.Compose(
            [
                T.Resize(config.image_size * 256 // 224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size=config.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        dataset_train = datasets.ImageFolder(os.path.join(config.data_path, "train"), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(config.data_path, "val"), transform=transform_val)
    xm.master_print(f"dataset_train = {dataset_train}\n")
    xm.master_print(f"dataset_val = {dataset_val}\n")
    # building training and validation data loaders
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=local_batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    data_loader_train_sampler = data_loader_train.sampler
    data_loader_train = pl.MpDeviceLoader(data_loader_train, device)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=world_size, rank=rank, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=local_batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    data_loader_val = pl.MpDeviceLoader(data_loader_val, device)

    # define the model
    model = getattr(models, config.model)(config=config, num_classes=NUM_CLASSES)
    model.to(device)
    xm.master_print(f"model = {model}\n")
    utils.broadcast_xla_master_model_param(model)
    if config.half_precision:
        model.to(torch.bfloat16)
    # define the optimizer
    scaled_lr = config.learning_rate * config.batch_size / 256.0
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=scaled_lr, weight_decay=config.weight_decay, momentum=config.momentum, nesterov=True
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=scaled_lr, weight_decay=config.weight_decay, betas=(config.adamw_b1, config.adamw_b2)
        )
    else:
        raise Exception(f"Invalid optimizer type: {config.optimizer}")
    steps_per_epoch = len(dataset_train) // config.batch_size
    lr_scheduler = utils.get_warmup_cosine_scheduler(
        optimizer,
        warmup_iteration=steps_per_epoch * config.warmup_epochs,
        max_iteration=steps_per_epoch * config.num_epochs,
    )
    xm.master_print(f"optimizer = {optimizer}\n")

    server = xp.start_server(9012)  # capture TPU profile
    profiler = multiprocessing.Process(target=capture_profile, args=(config.output_dir,))
    model.train()
    xm.master_print("Initial compilation, this might take some minutes...")
    start_time = time.time()
    train_metrics_last_t = time.time()
    for epoch in range(config.num_epochs):
        data_loader_train_sampler.set_epoch(epoch)
        for data_iter_step, (images, labels) in enumerate(data_loader_train):
            global_step = data_iter_step + len(data_loader_train) * epoch
            if global_step == 50 and rank == 0:
                profiler.start()  # capture profile starting from step 50th for 3000 ms
            with xp.StepTrace("training_step", step_num=global_step):
                optimizer.zero_grad()
                with xp.Trace("forward pass"):
                    logits = model(images.to(torch.bfloat16) if config.half_precision else images)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    accuracy = torch.mean((logits.argmax(-1) == labels).to(torch.float32))
                    # train_loss and train_accuracy for logging purpose
                    train_loss = xm.all_reduce(xm.REDUCE_SUM, loss, scale=1.0 / world_size)
                    train_accuracy = xm.all_reduce(xm.REDUCE_SUM, accuracy, scale=1.0 / world_size)
                with xp.Trace("backward pass"):
                    loss.backward()
                with xp.Trace("reduce gradients"):
                    xm.reduce_gradients(optimizer, pin_layout=config.pin_layout)
                with xp.Trace("optimizer update"):
                    optimizer.step()
                    lr_scheduler.step()

                if global_step == 0:
                    xm.add_step_closure(lambda: xm.master_print("Initial compilation completed."), args=())
                if (global_step + 1) % config.log_every_steps == 0:
                    new_last_t = time.time()
                    steps_per_second = config.log_every_steps / (new_last_t - train_metrics_last_t)
                    lr = optimizer.param_groups[0]["lr"]
                    xm.add_step_closure(
                        xla_logging, args=(global_step, train_loss, train_accuracy, lr, steps_per_second)
                    )
                    train_metrics_last_t = new_last_t

        if (epoch + 1) % config.eval_interval == 0 or epoch + 1 == config.num_epochs:
            utils.sync_batch_stats(model)
            accuracy_val, loss_val = eval_on_val(data_loader_val, model, device, config)
            xm.master_print(f"eval epoch: {epoch}, loss: {loss_val:.4f}, accuracy: {100*accuracy_val:.2f}")
            xm.master_print(
                f"{datetime.datetime.now().time()} [{global_step+1}] "
                f"eval_accuracy={accuracy_val:.8f}, eval_loss={loss_val:.8f}"
            )
        if (epoch + 1) % config.checkpoint_interval == 0 or epoch + 1 == config.num_epochs:
            utils.sync_batch_stats(model)
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            save_path = os.path.join(config.output_dir, f"checkpoint-{epoch}.pth")
            xm.save(ckpt, save_path, global_master=True)
            xm.master_print(f"Saved checkpoint at {save_path}")

    xm.master_print(f"Training time {datetime.timedelta(seconds=int(time.time() - start_time))}")


@torch.no_grad()
def eval_on_val(data_loader_val, model, device, config):
    model.eval()
    local_correct = torch.tensor(0, device=device)
    local_total = torch.tensor(0, device=device)
    local_loss = torch.tensor(0.0, device=device)
    for images, labels in data_loader_val:
        logits = model(images.to(torch.bfloat16) if config.half_precision else images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        local_correct += (logits.argmax(-1) == labels).sum()
        local_total += torch.tensor(labels.size(0), device=device)
        local_loss += loss
    correct = xm.all_reduce(xm.REDUCE_SUM, local_correct).item()
    total = xm.all_reduce(xm.REDUCE_SUM, local_total).item()
    loss_avg = xm.all_reduce(xm.REDUCE_SUM, local_loss).item()
    loss_avg = loss_avg / xm.xrt_world_size() / len(data_loader_val)
    accuracy = correct / total
    model.train()
    return accuracy, loss_avg


def xla_logging(global_step, train_loss, train_accuracy, lr, steps_per_second):
    xm.master_print(
        f"{datetime.datetime.now().time()} [{global_step+1}] "
        f"steps_per_second={steps_per_second:.8f}, "
        f"train_accuracy={train_accuracy.item():.8f}, train_learning_rate={lr:.8f}, "
        f"train_loss={train_loss.item():.8f}"
    )


def capture_profile(output_dir):
    xp.trace("localhost:9012", output_dir, duration_ms=3000)


def xla_main(index, config):
    main(config)


if __name__ == "__main__":
    parser = get_args_parser()
    config = parser.parse_args()
    xmp.spawn(xla_main, args=(config,))
