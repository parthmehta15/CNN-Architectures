import torch
import torch.nn as nn
import math


class InvertedResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__() # Just have to do this for all nn.Module classes


        #Depthwise Convolution
        self.spatial_mixing = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding =3,
                    stride = 1, groups = in_channels, bias=False),
            nn.BatchNorm2d(in_channels),

        )

        # Expand Ratio is like 4, so hidden_dim >> in_channels
        hidden_dim = in_channels * 4

        #Pointwise Convolution
        self.feature_mixing = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, padding =0,
                    stride = 1, bias=False),
            nn.GELU(),
        )

        self.bottleneck_channels = nn.Sequential(
             nn.Conv2d(hidden_dim,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
        )

    def forward(self, x):
        out = self.spatial_mixing(x)
        out = self.feature_mixing(out)
        out = self.bottleneck_channels(out)
        return x + out

class ConvNext(nn.Module):

    def __init__(self, num_classes= 7000):
        super().__init__()

        self.num_classes = num_classes

        """
        First couple of layers are special, just do them here.
        This is called the "stem". Usually, methods use it to downsample or twice.
        """
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            nn.BatchNorm2d(96),
        )


        self.stage_cfgs = [
            # expand_ratio, channels, # blocks, stride of first block
            [4,  96, 3, 1],
            [4,  192, 3, 1],
            [4,  384, 9, 1],
            [4,  768, 3, 1],

        ]

        in_channels = 96
        layers = []


        #BLOCK TYPE 1 - 3 TIMES
        for i in range(3):
            layers.append(InvertedResidualBlock(
                in_channels=96,
                out_channels=96))

        layers.append(nn.BatchNorm2d(96))
        layers.append(nn.Conv2d(96,192,kernel_size=2,stride=2))

        #BLOCK TYPE 2 - 3 TIMES
        for i in range(3):
            layers.append(InvertedResidualBlock(
                in_channels=192,
                out_channels=192))

        layers.append(nn.BatchNorm2d(192))
        layers.append(nn.Conv2d(192,384,kernel_size=2,stride=2))

        #BLOCK TYPE 3 - 9 TIMES
        for i in range(9):
            layers.append(InvertedResidualBlock(
                in_channels=384,
                out_channels=384))

        layers.append(nn.BatchNorm2d(384))
        layers.append(nn.Conv2d(384,768,kernel_size=2,stride=2))

        #BLOCK TYPE 4 - 3 TIMES
        for i in range(3):
            layers.append(InvertedResidualBlock(
                in_channels=768,
                out_channels=768))



        self.layers = nn.Sequential(*layers)
        self.mid_cls_layer = nn.Sequential(

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        self.final_cls_layer = nn.Sequential(nn.Linear(768,num_classes),)



    def forward(self, x,return_feats=False):
        out = self.stem(x)
        out = self.layers(out)

        feats = self.mid_cls_layer(out)

        out = self.final_cls_layer(feats)


        if return_feats:
            return feats
        else:
            return out
