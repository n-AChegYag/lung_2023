import torch
from torch import nn

from .SE_layers import FastSmoothSeNormConv3d, RESseNormConv3d, UpConv
import torch.nn.functional as F

class MyNet(nn.Module):

    def __init__(self, in_channels, n_filters, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.reduction = reduction
        
        self.block_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.classifer = nn.Sequential(
            nn.AdaptiveAvgPool3d((2,3,3)),
            nn.Flatten(),
            nn.Linear(18 * 8 * n_filters, 128),
            nn.Linear(18 * 8 * n_filters, 1)
        )

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.block_3_1_right = FastSmoothSeNormConv3d(8 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.vision_3 = UpConv(4 * n_filters, n_filters, reduction, scale=4)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.block_2_1_right = FastSmoothSeNormConv3d(4 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.vision_2 = UpConv(2 * n_filters, n_filters, reduction, scale=2)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.block_1_1_right = FastSmoothSeNormConv3d(2 * n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_3_left(self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0))))
        ds2 = self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1))))
        x = self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2))))
        class_logit = self.classifer(x)
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], dim=1)))
        sv3 = self.vision_3(x)
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], dim=1)))
        sv2 = self.vision_2(x)
        x = self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], dim=1))
        x = x + sv2 + sv3
        x = self.block_1_2_right(x)
        x = self.conv1x1(x)

        # return torch.sigmoid(class_logit), torch.sigmoid(x)
        return class_logit, x


class MyClassifer(nn.Module):
    
    def __init__(self, n_filters, n_radiomics, class_num):
        super().__init__()
        self.class_num = class_num
        self.n_filters = n_filters
        self.n_radiomics = n_radiomics
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool3d((2,3,3)),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(18 * 8 * self.n_filters, 128)
        self.fc2 = nn.Linear(128 + n_radiomics, self.class_num)
    
    def forward(self, x, radiomics_feats):
        x = self.flatten(x)
        x = self.fc1(x)
        if self.n_radiomics != 0:
            x = self.fc2(torch.cat([x, radiomics_feats], dim=1))
        else:
            x = self.fc2(x)
        return x


class MyEncoder(nn.Module):

    def __init__(self, in_channels, n_filters, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.reduction = reduction
        
        self.block_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_3_left(self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0))))
        ds2 = self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1))))
        x = self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2))))

        return x, ds0, ds1, ds2
    
    
class MyDecoder(nn.Module):

    def __init__(self, in_channels, n_filters, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.reduction = reduction

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.block_3_1_right = FastSmoothSeNormConv3d(8 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.vision_3 = UpConv(4 * n_filters, n_filters, reduction, scale=4)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.block_2_1_right = FastSmoothSeNormConv3d(4 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.vision_2 = UpConv(2 * n_filters, n_filters, reduction, scale=2)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.block_1_1_right = FastSmoothSeNormConv3d(2 * n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, ds0, ds1, ds2):

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], dim=1)))
        sv3 = self.vision_3(x)
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], dim=1)))
        sv2 = self.vision_2(x)
        x = self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], dim=1))
        x = x + sv2 + sv3
        x = self.block_1_2_right(x)
        x = self.conv1x1(x)

        return x