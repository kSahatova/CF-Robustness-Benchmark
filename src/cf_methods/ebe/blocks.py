from torch import nn



class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization"""
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)
        
    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(y).view(-1, self.num_features, 1, 1)
        return gamma * out + beta