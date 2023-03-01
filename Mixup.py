from torch.distributions import Uniform
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


# AUGMENTATIONS ON RAW AUDIO
# MIXUP
class Mixup(nn.Module):
    def __init__(self, alpha, num_classes):
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.beta_dist = Uniform(alpha, 1)  # if uniform use values from 0.5 upwards or Beta(alpha, alpa)

    @classmethod
    def mix(cls, data: torch.Tensor, gamma, indices: torch.Tensor) -> torch.Tensor:
        assert data.shape[0] == indices.shape[0], 'Requires same number of samples'
        gamma_ = gamma.view(gamma.shape[0], *(1 for _ in range(len(data.shape) - 1)))
        return data * gamma_ + (1.0 - gamma_) * data[indices]

    def forward(self, data, labels):
        gamma = self.beta_dist.sample((data.shape[0],))
        indices = torch.randperm(data.size(0), device=data.device, dtype=torch.long)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=self.num_classes) if len(
            labels.shape) <= 1 else labels
        mixedup_data, mixedup_labels = self.mix(data, gamma, indices), self.mix(one_hot_labels, gamma, indices)
        return mixedup_data, mixedup_labels


def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    input = torch.log(torch.nn.functional.softmax(input, dim=1).clamp(1e-5, 1))
    # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def onehot(targets, num_classes):
    """Origin: https://github.com/moskomule/mixup.pytorch
    convert index tensor into onehot tensor
    :param targets: index tensor
    :param num_classes: number of classes
    """
    assert isinstance(targets, torch.LongTensor)
    return torch.zeros(targets.size()[0], num_classes).scatter_(1, targets.view(-1, 1), 1)


def mixup(inputs, targets, num_classes, alpha=2):
    s = inputs.size()[0]
    weight = torch.Tensor(np.random.beta(alpha, alpha, s))
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index, :, :, :]
    y1, y2 = onehot(targets, num_classes), onehot(targets[index,], num_classes)
    weight = weight.view(s, 1, 1, 1)
    inputs = weight * x1 + (1 - weight) * x2
    weight = weight.view(s, 1)
    targets = weight * y1 + (1 - weight) * y2
    return inputs, targets
