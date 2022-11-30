from functools import partial

from timm.models.resnet import ResNet, BasicBlock, Bottleneck


def _get_resnet(config, *args, **kwargs):
    # remove the config argument (not needed in ResNet)
    return ResNet(*args, **kwargs)


# ResNet V1 models
ResNet18 = partial(_get_resnet, layers=[2, 2, 2, 2], block=BasicBlock)
ResNet34 = partial(_get_resnet, layers=[3, 4, 6, 3], block=BasicBlock)
ResNet50 = partial(_get_resnet, layers=[3, 4, 6, 3], block=Bottleneck)
ResNet101 = partial(_get_resnet, layers=[3, 4, 23, 3], block=Bottleneck)
ResNet152 = partial(_get_resnet, layers=[3, 8, 36, 3], block=Bottleneck)
ResNet200 = partial(_get_resnet, layers=[3, 24, 36, 3], block=Bottleneck)
