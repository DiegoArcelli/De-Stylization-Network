import torch
from torch import nn
from torch import Tensor
from torch.functional import Tensor
from torch.nn import Module


class AdaptiveInstanceNormalization(Module):


    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features


    def forward(self, content_feature: Tensor, style_feature: Tensor) -> Tensor:
        n, c = content_feature.shape[:2]
        assert(c == self.num_features)
        assert((n, 2*c) == style_feature.shape)
        size = content_feature.size()
        style_mean, style_std = style_feature[:, :c].view(n, c, 1, 1), style_feature[:, c:].view(n, c, 1, 1)
        feature_mean, feature_std = self.mean(content_feature), self.standard_deviation(content_feature)
        standardized_feature = (content_feature - feature_mean.expand(size)) / feature_std.expand(size)
        res = standardized_feature * style_std.expand(size) + style_mean.expand(size)
        return res


    def mean(self, x: Tensor) -> Tensor:
        n, c = x.shape[:2]
        return x.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)


    def standard_deviation(self, x: Tensor, eps=1e-5) -> Tensor:
        n, c = x.shape[:2]
        var = x.view(n, c, -1).var(dim=2) + eps
        return var.sqrt().view(n, c, 1, 1)