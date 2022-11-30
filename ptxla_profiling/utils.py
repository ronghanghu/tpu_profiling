from itertools import chain
import math
import random

import torch
import torch_xla.core.xla_model as xm


def broadcast_xla_master_model_param(model):
    """
    Broadcast the model parameters from master process to other processes
    """
    parameters_and_buffers = []
    is_master = xm.is_master_ordinal(local=False)
    for p in chain(model.parameters(), model.buffers()):
        # Set all params in non-master devices to zero so that all_reduce is
        # equivalent to broadcasting parameters from master to other devices.
        scale = 1 if is_master else 0
        scale = torch.tensor(scale, dtype=p.data.dtype, device=p.data.device)
        p.data.mul_(scale)
        parameters_and_buffers.append(p.data)
    xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
    xm.mark_step()


def sync_batch_stats(model):
    """ Average the model buffers across ranks """
    buffers = [p.data for p in model.buffers()]
    xm.all_reduce(xm.REDUCE_SUM, buffers, scale=1.0 / xm.xrt_world_size())
    xm.mark_step()


def get_warmup_cosine_scheduler(optimizer, warmup_iteration, max_iteration):
    def _warmup_cosine(step):
        if step < warmup_iteration:
            lr_ratio = step * 1.0 / warmup_iteration
        else:
            where = (step - warmup_iteration) * 1.0 / (max_iteration - warmup_iteration)
            lr_ratio = 0.5 * (1 + math.cos(math.pi * where))

        return lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine)


class FakeDataset:
    def __init__(self, num_samples, image_size, num_classes):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __getitem__(self, idx):
        image = torch.zeros(3, self.image_size, self.image_size)
        label = random.randrange(self.num_classes)
        return image, label

    def __len__(self):
        return self.num_samples
