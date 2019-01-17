import torch
import torch.nn as nn

from .fcn_mask_head import FCNMaskHead
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class PANetMaskHead(FCNMaskHead):

    def __init__(self,
                 num_inputs=4,
                 fc_position=3,
                 num_fc_convs=2,
                 *args,
                 **kwargs):
        super(PANetMaskHead, self).__init__(*args, **kwargs)
        self.num_inputs = num_inputs
        self.fc_position = fc_position
        self.num_fc_convs = num_fc_convs
        assert self.num_inputs >= 2
        assert self.num_convs >= 1
        assert self.fc_position <= self.num_convs
        assert self.num_fc_convs >= 2
        fc_conv_channles = self.conv_out_channels // 2

        self.fc_convs = nn.ModuleList()
        for i in range(self.num_fc_convs - 1):
            self.fc_convs.append(
                ConvModule(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    normalize=self.normalize,
                    bias=self.with_bias))
        # reduce feature map channels
        self.fc_convs.append(
            ConvModule(
                self.conv_out_channels,
                fc_conv_channles,
                3,
                padding=1,
                normalize=self.normalize,
                bias=self.with_bias))
        self.convs[0] = nn.ModuleList()
        for i in range(num_inputs):
            self.convs[0].append(
                ConvModule(
                    self.in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    normalize=self.normalize,
                    bias=self.with_bias))

        self.fc = nn.Linear(
            fc_conv_channles * self.roi_feat_size * self.roi_feat_size,
            self.roi_feat_size * self.roi_feat_size * 4)

    def init_weights(self):
        super(PANetMaskHead, self).init_weights()
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        assert len(x) == self.num_inputs
        x = [conv(x_) for x_, conv in zip(x, self.convs[0])]

        x, _ = torch.stack(x).max(dim=0)

        for i in range(1, self.fc_position):
            x = self.convs[i](x)
        y = x
        for i in range(self.fc_position, self.num_convs):
            x = self.convs[i](x)
        x = self.upsample(x)
        if self.upsample_method == 'deconv':
            x = self.relu(x)
        mask_pred = self.conv_logits(x)

        for fc_conv in self.fc_convs:
            y = fc_conv(y)
        x_fc = self.fc(y.view(x.size(0), -1))
        fc_pred = x_fc.view(
            mask_pred.size(0), 1, mask_pred.size(2), mask_pred.size(3))
        mask_pred += fc_pred

        return mask_pred
