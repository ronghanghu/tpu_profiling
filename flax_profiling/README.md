## JAX (FLAX) profiling

### Setup

1. Allocate a TPU VM (e.g. v3-128) with `tpu-vm-base` TPU VM runtime. Then install the following dependencies on each VM host.
```bash
sudo pip3 install ml-collections==0.1.0
sudo pip3 install optax==0.1.2
sudo pip3 install tensorflow-datasets==4.5.2
sudo pip3 install clu==0.0.7
sudo pip3 install absl-py==1.1.0

sudo pip3 install jax[tpu]==0.3.25 jaxlib==0.3.25 flax==0.6.2 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

2. Clone this repo to a local directory, e.g. `/checkpoint/ronghanghu/workspace/tpu_profiling`.

3. Follow [`imagenet2012` dataset guide](https://www.tensorflow.org/datasets/catalog/imagenet2012) in `tensorflow_datasets` to set up the ImageNet-1k dataset and set up the `TFDS_DATA_DIR` environment variable to point to your `tensorflow_datasets` storage location (e.g. `gs://ronghanghu_storage/datasets/tensorflow_datasets`).

As an alterative, you can pass `--config.use_fake_data=True` below to run profiling with fake data.

### Running profiling

Example profiling ResNet50 on v3-128 with batch size 8192 and bfloat16 parameters:
```bash
TPU_NAME=rh-128-x
ZONE=europe-west4-a

CODE_DIR=/checkpoint/ronghanghu/workspace/tpu_profiling/flax_profiling
TFDS_DATA_DIR=gs://ronghanghu_storage/datasets/tensorflow_datasets
SAVE_DIR=/checkpoint/ronghanghu/ptxla_vs_jax_tpu_profiling/flax/resnet50_bs8192_bfloat16_img224_sgd_v3-128

MODEL=ResNet50
BATCH_SIZE=8192
EPOCH=100

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # workaround for permission issue
cd ${HOME} && gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --worker all --command "
cd ${CODE_DIR}

TFDS_DATA_DIR=${TFDS_DATA_DIR} \
python3 main.py \
  --workdir=${SAVE_DIR} \
  --config=configs/tpu_profile.py \
  --config.model=${MODEL} \
  --config.batch_size=${BATCH_SIZE} \
  --config.num_epochs=${EPOCH} \
  --config.half_precision=True
" 2>&1 | tee ${SAVE_DIR}/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
```

It will save the captured TPU profile under `${SAVE_DIR}`, which can be viewed from TensorBoard.
