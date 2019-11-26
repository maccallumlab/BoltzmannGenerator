from .base import Transform
import torch


class NoiseTransform(Transform):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, inputs, context=None):
        noise = torch.normal(0, self.std, size=inputs.shape).to(inputs.device)
        return inputs + noise, torch.zeros(inputs.shape[0], device=inputs.device)

    def inverse(self, inputs, context=None):
        noise = torch.normal(0, self.std, size=inputs.shape).to(inputs.device)
        return inputs + noise, torch.zeros(inputs.shape[0], device=inputs.device)


class TwoStageComposite(Transform):
    def __init__(self, stage1, stage2):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def forward(self, inputs, context=None):
        x, jac = self.stage1.forward(inputs, context)
        x, jac2 = self.stage2.forward(x, context)
        return x, jac + jac2

    def inverse(self, inputs, context=None):
        x, jac = self.stage2.inverse(inputs, context)
        x, jac2 = self.stage1.inverse(x, context)
        return x, jac + jac2

    def stage1_forward(self, inputs, context=None):
        return self.stage1.forward(inputs, context)

    def stage1_inverse(self, inputs, context=None):
        return self.stage1.inverse(inputs, context)

    def stage2_forward(self, inputs, context=None):
        return self.stage2.forward(inputs, context)

    def stage2_inverse(self, inputs, context=None):
        return self.stage2.inverse(inputs, context)

