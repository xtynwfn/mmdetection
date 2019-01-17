import torch
import torch.nn as nn

from .convfc_bbox_head import ConvFCBBoxHead
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class PANetBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_inputs=4, *args, **kwargs):
        super(PANetBBoxHead, self).__init__(*args, **kwargs)
        self.num_inputs = num_inputs
        assert self.num_inputs >= 2
        assert self.num_shared_convs >= 1

        self.shared_convs[0] = nn.ModuleList()
        for i in range(num_inputs):
            self.shared_convs[0].append(
                ConvModule(
                    self.in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    normalize=self.normalize,
                    bias=self.with_bias))

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) == self.num_inputs
        x = [
            shared_conv(x_) for x_, shared_conv in zip(x, self.shared_convs[0])
        ]
        x, _ = torch.stack(x).max(dim=0)

        # shared part
        for conv in self.shared_convs[1:]:
            x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
