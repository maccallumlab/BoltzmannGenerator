from .base import Transform
import torch


class ForwardNoiseTransform(Transform):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, inputs, context=None):
        if self.training:
            noise = torch.normal(0, self.std, size=inputs.shape).to(inputs.device)
            return inputs + noise, torch.zeros(inputs.shape[0], device=inputs.device)
        else:
            return inputs, torch.zeros(inputs.shape[0], device=inputs.device)

    def inverse(self, inputs, context=None):
        return inputs, torch.zeros(inputs.shape[0], device=inputs.device)