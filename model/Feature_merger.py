import torch
import torch.nn as nn
import torch.nn.functional as F


class Feature_merger(nn.Module):
    def __init__(self, input_dim, num_patches, downsample_dim):
        super().__init__()

        self.output_dim = 2*downsample_dim
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(input_dim, 128)

        self.fuse1 = torch.nn.Linear(128*num_patches**2, downsample_dim)
        self.fuse2 = torch.nn.Linear(128*num_patches**2, downsample_dim)

        self.activation1 = torch.nn.Tanh()
        self.activation2 = torch.nn.Tanh()

    
    def _unpack_pair(self, feature_list, feature2=None):
        if feature2 is not None:
            return feature_list, feature2
        if isinstance(feature_list, (list, tuple)) and len(feature_list) == 2:
            return feature_list[0], feature_list[1]
        raise ValueError("Feature_merger expects two feature tensors.")

    def forward(self, feature_list, feature2=None):
        feature1, feature2 = self._unpack_pair(feature_list, feature2)
        feature_transformed_1 = self.fuse1(torch.flatten(self.lin1(feature1), start_dim=1))
        feature_transformed_2 = self.fuse2(torch.flatten(self.lin2(feature2), start_dim=1))
        merged = torch.concatenate((feature_transformed_1, feature_transformed_2), dim = -1)
        normalized_merged = F.normalize(merged, p = 2, dim = 1)
        
        return normalized_merged


