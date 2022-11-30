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

# Copyright 2021 The Flax Authors.
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
"""Hyperparameter configuration to run the example on TPUs."""

import ml_collections


def get_config():
    """Get the hyperparameter configuration to train on TPUs."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = "ResNet50"
    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "imagenet2012:5.*.*"
    config.cache = True
    config.use_fake_data = False
    config.image_size = 224

    config.optimizer = "sgd"
    config.learning_rate = 0.1
    config.warmup_epochs = 5.0
    config.momentum = 0.9
    config.adamw_b1 = 0.9
    config.adamw_b2 = 0.999
    config.weight_decay = 0.0001

    config.num_epochs = 100
    config.log_every_steps = 20
    config.checkpoint_interval = 10
    config.eval_interval = 10

    # Consider setting the batch size to max(tpu_chips * 256, 8 * 1024) if you
    # train on a larger pod slice.
    config.batch_size = 1024
    config.half_precision = False  # run with full precision by default

    return config
