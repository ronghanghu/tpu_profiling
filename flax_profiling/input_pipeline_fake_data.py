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

"""Fake ImageNet input pipeline.
"""

import jax
import tensorflow as tf


def create_split(dataset_builder, batch_size, train, dtype=tf.float32, image_size=224, cache=False):
    num_classes = dataset_builder["num_classes"]
    split_size = dataset_builder["num_train" if train else "num_val"] // jax.process_count()

    def decode_example(_):
        image = tf.zeros([image_size, image_size, 3], dtype=dtype)
        label = tf.random.uniform(shape=[1], maxval=num_classes, dtype=tf.int64)
        return {"image": image, "label": label}

    ds = tf.data.Dataset.from_tensor_slices(list(range(split_size)))
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(16 * batch_size, seed=0)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(10)

    return ds
