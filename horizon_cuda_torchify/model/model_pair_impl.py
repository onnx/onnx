import torch
import torch.nn as nn

from cuda_op import custom_add


class CustomAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return custom_add(x, y)

    @staticmethod
    def symbolic(g, x, y):
        return g.op("custom_domain::CustomAdd", x, y)


class ModelWithCuda(nn.Module):
    def custom_op_call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return CustomAddFunction.apply(x, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.custom_op_call(x, y)


class ModelTorch(nn.Module):
    def custom_op_call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # torch化替换点：此处与 ModelWithCuda 保持相同调用形式，仅将实现替换为标准算子。
        return x + y

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.custom_op_call(x, y)
