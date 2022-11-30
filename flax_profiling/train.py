# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import datetime
import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import input_pipeline_fake_data
import models


NUM_CLASSES = 1000


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args, train=False)

    variables = init({"params": key}, jnp.ones(input_shape, model.dtype))
    return variables


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def create_learning_rate_fn(config: ml_collections.ConfigDict, base_learning_rate: float, steps_per_epoch: int):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=base_learning_rate, transition_steps=config.warmup_epochs * steps_per_epoch
    )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[config.warmup_epochs * steps_per_epoch]
    )
    return schedule_fn


def train_step(state, batch, learning_rate_fn, config):
    """Perform a single training step."""

    _, new_rng = jax.random.split(state.rng)
    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    dropout_rng = jax.random.fold_in(state.rng, jax.lax.axis_index("batch"))

    def loss_fn(params):
        """loss function used for training."""
        logits, new_model_state = state.apply_fn(
            {"params": params, **state.variables},
            batch["image"],
            mutable=[k for k in state.variables],
            rngs=dict(dropout=dropout_rng),
            train=True,
        )
        loss = cross_entropy_loss(logits, batch["label"])
        if config.optimizer == "sgd":
            weight_penalty_params = jax.tree_util.tree_leaves(params)
            weight_decay = config.weight_decay
            weight_l2 = sum(jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1)
            weight_penalty = weight_decay * 0.5 * weight_l2
            loss = loss + weight_penalty
        return loss, (new_model_state, logits)

    step = state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name="batch")

    new_variables, logits = aux[1]
    metrics = compute_metrics(logits, batch["label"])
    metrics["learning_rate"] = lr

    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(
        step=state.step + 1, params=new_params, opt_state=new_opt_state, variables=new_variables, rng=new_rng,
    )

    return new_state, metrics


def eval_step(state, batch):
    variables = {"params": state.params, **state.variables}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    return compute_metrics(logits, batch["label"])


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train, config):
    if config.use_fake_data:
        ds = input_pipeline_fake_data.create_split(
            dataset_builder, batch_size, image_size=image_size, dtype=dtype, train=train, cache=config.cache
        )
    else:
        ds = input_pipeline.create_split(
            dataset_builder, batch_size, image_size=image_size, dtype=dtype, train=train, cache=config.cache
        )
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


class TrainState(train_state.TrainState):
    rng: Any
    variables: flax.core.FrozenDict[str, Any]


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3, overwrite=True)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    if "batch_stats" in state.variables:
        mean_batch_stats = cross_replica_mean(state.variables["batch_stats"])
        new_variables = state.variables.copy({"batch_stats": mean_batch_stats})
        return state.replace(variables=new_variables)

    return state


def create_train_state(rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn):
    """Create initial training state."""
    # split rng for init and for state
    rng_init, rng_state = jax.random.split(rng)

    variables = initialized(rng_init, image_size, model)
    variables_states, params = variables.pop("params")
    if config.optimizer == "sgd":
        # for SGD, the weight decay will be added as an L2 regularization term in the loss
        tx = optax.sgd(learning_rate=learning_rate_fn, momentum=config.momentum, nesterov=True)
    elif config.optimizer == "adamw":
        tx = optax.adamw(
            learning_rate=learning_rate_fn, b1=config.adamw_b1, b2=config.adamw_b2, weight_decay=config.weight_decay
        )
    else:
        raise Exception(f"Invalid optimizer type: {config.optimizer}")
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, rng=rng_state, variables=variables_states)
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      Final TrainState.
    """

    writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)

    rng = random.PRNGKey(0)
    image_size = config.image_size
    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    local_batch_size = config.batch_size // jax.process_count()
    input_dtype = tf.bfloat16 if config.half_precision else tf.float32

    if config.use_fake_data:
        dataset_builder = {"num_train": 1281167, "num_val": 50000, "num_classes": 1000}
        steps_per_epoch = dataset_builder["num_train"] // config.batch_size
        steps_per_eval = dataset_builder["num_val"] // config.batch_size
    else:
        dataset_builder = tfds.builder(config.dataset)
        steps_per_epoch = dataset_builder.info.splits["train"].num_examples // config.batch_size
        steps_per_eval = dataset_builder.info.splits["validation"].num_examples // config.batch_size
    train_iter = create_input_iter(
        dataset_builder, local_batch_size, image_size, input_dtype, train=True, config=config
    )
    eval_iter = create_input_iter(
        dataset_builder, local_batch_size, image_size, input_dtype, train=False, config=config
    )
    num_steps = int(steps_per_epoch * config.num_epochs)

    model_cls = getattr(models, config.model)
    model_dtype = jnp.bfloat16 if config.half_precision else jnp.float32
    model = model_cls(num_classes=NUM_CLASSES, dtype=model_dtype)
    base_learning_rate = config.learning_rate * config.batch_size / 256.0
    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)
    state = create_train_state(rng, config, model, image_size, learning_rate_fn)
    # state = checkpoints.restore_checkpoint(workdir, state)  # skip checkpoint restoration in profiling
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn, config=config), axis_name="batch"
    )
    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(first_profile=50, profile_duration_ms=3000, every_secs=None, logdir=workdir)]
    logging.info("Initial compilation, this might take some minutes...")
    start_time = time.time()
    train_metrics_last_t = time.time()
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, metrics = p_train_step(state, batch)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info("Initial compilation completed.")

        if config.get("log_every_steps"):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train_{k}": v for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary["steps_per_second"] = config.log_every_steps / (time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % (steps_per_epoch * config.eval_interval) == 0 or step + 1 == num_steps:
            epoch = step // steps_per_epoch
            eval_metrics = []

            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info(
                "eval epoch: %d, loss: %.4f, accuracy: %.2f", epoch, summary["loss"], summary["accuracy"] * 100
            )
            writer.write_scalars(step + 1, {f"eval_{key}": val for key, val in summary.items()})
            writer.flush()
        if (step + 1) % (steps_per_epoch * config.checkpoint_interval) == 0 or step + 1 == num_steps:
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    logging.info(f"Training time {datetime.timedelta(seconds=int(time.time() - start_time))}")

    return state
