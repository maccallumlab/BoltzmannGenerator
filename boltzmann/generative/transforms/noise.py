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


class SelectiveMiddleComposite(Transform):
    def __init__(self, head, middle, tail):
        super().__init__()
        self.head = head
        self.middle = middle
        self.tail = tail

    def forward(self, inputs, context=None, use_middle=True):
        x, jac_head = self.head.forward(inputs, context)
        jac_total = jac_head

        if use_middle:
            x, jac_middle = self.middle.forward(x, context)
            jac_total = jac_total + jac_middle

        x, jac_tail = self.tail.forward(x, context)
        jac_total = jac_total + jac_tail

        return x, jac_total

    def inverse(self, inputs, context=None, use_middle=True):
        x, jac_tail = self.tail.inverse(inputs, context)
        jac_total = jac_tail

        if use_middle:
            x, jac_middle = self.middle.inverse(x, context)
            jac_total = jac_total + jac_middle

        x, jac_head = self.head.inverse(x, context)
        jac_total = jac_total + jac_head

        return x, jac_total

