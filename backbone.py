import torch
import torch.nn as nn
from forward_process import *
from dataset import *
from dataset import *
import timm
from torch import Tensor, nn
from typing import Callable, List, Tuple, Union



class Feature_extractor(nn.Module):
    def __init__(self , config) -> None:
        super().__init__()

        self.config = config
        self.input_size = config.data.image_size
        self.backbone = config.model.backbone
        if self.backbone in ["cait_m48_448"]:
            self.feature_extractor = timm.create_model((self.backbone), pretrained=True)
            channels = [768]
            scales = [16]


        elif self.backbone in ["resnet18", "wide_resnet50_2"]:
            self.feature_extractor =  timm.create_model(
                        self.backbone,
                        pretrained=True,
                        features_only=True,
                        out_indices=[1],
                    )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()
            self.norms = nn.ModuleList()


            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(config.data.image_size / scale), int(config.data.image_size / scale)],
                        elementwise_affine=True,
                    )
                )


        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.feature_extractor.eval()
        if self.backbone in ["cait_m48_448"]:

            feature = self.feature_extractor.patch_embed(x)
            feature = feature + self.feature_extractor.pos_embed
            feature = self.feature_extractor.pos_drop(feature)
            for i in range(41):  # paper Table 6. Block Index = 40
                feature = self.feature_extractor.blocks[i](feature)
            batch_size, _, num_channels = feature.shape
            feature = self.feature_extractor.norm(feature)
            feature = feature.permute(0, 2, 1)
            feature = feature.reshape(batch_size, num_channels, self.input_size // 16, self.input_size // 16)
            return feature
        elif self.backbone in ["resnet18", "wide_resnet50_2"]:
            features = self.feature_extractor(x)

            features = [self.norms[i](feature) for i, feature in enumerate(features)]


            embeddings = features[0]
            for feature in features[1:]:
                layer_embedding = feature
                layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode='nearest')
                embeddings = torch.cat((embeddings,layer_embedding),1)

            return embeddings
        