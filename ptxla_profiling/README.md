## PyTorch/XLA profiling

### Setup

1. Allocate a TPU VM (e.g. v3-128) with `tpu-vm-pt-1.12` TPU VM runtime. Then install the following dependencies on each VM host.
```bash
PTXLA_NIGHTLY_DATE=20221117
LIBTPU_NIGHTLY_DATE=20221017

# to resolve tensorflow import issue in https://github.com/pytorch/xla/issues/3786
sudo pip3 uninstall -y tensorflow
sudo pip3 uninstall -y tensorflow  # do it twice in case of any duplicated installation
sudo pip3 uninstall -y tf-nightly
sudo pip3 install tensorflow-cpu==2.9.1

# torch, torchvision and torch_xla
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly+${PTXLA_NIGHTLY_DATE}-cp38-cp38-linux_x86_64.whl
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly+${PTXLA_NIGHTLY_DATE}-cp38-cp38-linux_x86_64.whl
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly+${PTXLA_NIGHTLY_DATE}-cp38-cp38-linux_x86_64.whl
# libtpu
sudo pip3 install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.dev${LIBTPU_NIGHTLY_DATE}-py3-none-any.whl

# for ResNet and ViT models
sudo pip3 install timm==0.4.12
```

2. Clone this repo to a local directory, e.g. `/checkpoint/ronghanghu/workspace/tpu_profiling`.

3. Download [ImageNet-1k](https://image-net.org/) to a shared directory (e.g. /dataset02/imagenet-1k/), which should have the following structure (the validation images moved to labeled subfolders, following the [PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet#requirements)).
```
/dataset02/imagenet-1k/
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```

As an alterative, you can pass `--use_fake_data` below to run profiling with fake data.

### Running profiling

Example profiling ResNet50 on v3-128 with batch size 8192 and bfloat16 parameters:
```bash
TPU_NAME=rh-128-y
ZONE=europe-west4-a

CODE_DIR=/checkpoint/ronghanghu/workspace/tpu_profiling/ptxla_profiling
DATA_PATH=/datasets02/imagenet-1k/
SAVE_DIR=/checkpoint/ronghanghu/ptxla_vs_jax_tpu_profiling/ptxla/resnet50_bs8192_bfloat16_img224_sgd_v3-128

MODEL=ResNet50
BATCH_SIZE=8192
EPOCH=100

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # workaround for permission issue
cd ${HOME} && gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --worker all --command "
cd ${CODE_DIR}

XLA_HLO_DEBUG=1 PJRT_DEVICE=TPU PYTHONUNBUFFERED=1 \
python3 main.py \
    --output_dir ${SAVE_DIR} --data_path ${DATA_PATH} \
    --model ${MODEL} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${EPOCH} \
    --half_precision
" 2>&1 | tee ${SAVE_DIR}/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
```

It will save the captured TPU profile under `${SAVE_DIR}`, which can be viewed from TensorBoard.
