"""
Libra Models - Quantized Linear Projection Layers
Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)

Provides drop-in replacements for torch.nn.Linear:
1. QuantizedLinearINT8: Symmetric INT8 weights with per-channel scaling (4x memory reduction)
2. QuantizedLinearINT4: Symmetric INT4 packed weights (8x memory reduction)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.quantization.quant_core import (
    dequantize_symmetric,
    pack_int4,
    quantize_symmetric,
    unpack_int4,
)


class QuantizedLinearINT8(nn.Module):
    """
    Linear layer with symmetrically quantized 8-bit integer weights.
    Weights are stored as int8 tensors with per-channel (per-output-neuron) float32 scaling factors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Stored quantized weight: int8 of shape (out_features, in_features)
        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        # Per-channel scale: float32 of shape (out_features, 1)
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear, per_channel: bool = True) -> QuantizedLinearINT8:
        """Constructs a QuantizedLinearINT8 module from an existing trained nn.Linear layer."""
        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )

        w = linear.weight.detach()
        w_q, scale = quantize_symmetric(w, bits=8, per_channel=per_channel, dim=0)

        qlinear.weight_q.copy_(w_q)
        qlinear.weight_scale.copy_(scale)

        if linear.bias is not None:
            qlinear.bias.data.copy_(linear.bias.detach())

        return qlinear

    def dequantize_weight(self) -> torch.Tensor:
        """Reconstructs the float32 weight matrix from int8 storage."""
        return dequantize_symmetric(self.weight_q, self.weight_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly weight dequantization.
        Y = X @ W_dequantized.T + bias
        """
        w_dequant = self.dequantize_weight()
        return F.linear(x, w_dequant, self.bias)

    def memory_bytes(self) -> int:
        """Calculates total memory occupied by weights and scaling parameters in bytes."""
        total = self.weight_q.nelement() * 1  # 1 byte per int8 element
        total += self.weight_scale.nelement() * 4  # 4 bytes per float32 scale
        if self.bias is not None:
            total += self.bias.nelement() * 4
        return total


class QuantizedLinearINT4(nn.Module):
    """
    Linear layer with symmetrically quantized 4-bit integer weights.
    Two 4-bit weights are packed into a single uint8 byte, yielding 8x memory reduction over FP32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_features % 2 != 0:
            raise ValueError(f"in_features ({in_features}) must be even for INT4 packing")

        self.in_features = in_features
        self.out_features = out_features

        # Stored packed weight: uint8 of shape (out_features, in_features // 2)
        self.register_buffer(
            "weight_packed",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
        )
        # Per-channel scale: float32 of shape (out_features, 1)
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear, per_channel: bool = True) -> QuantizedLinearINT4:
        """Constructs a QuantizedLinearINT4 module from an existing trained nn.Linear layer."""
        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )

        w = linear.weight.detach()
        # Quantize to 4-bit signed [-8, 7]
        w_q, scale = quantize_symmetric(w, bits=4, per_channel=per_channel, dim=0)

        # Pack into uint8
        packed = pack_int4(w_q)

        qlinear.weight_packed.copy_(packed)
        qlinear.weight_scale.copy_(scale)

        if linear.bias is not None:
            qlinear.bias.data.copy_(linear.bias.detach())

        return qlinear

    def dequantize_weight(self) -> torch.Tensor:
        """Unpacks and dequantizes uint8 buffer into float32 weight matrix."""
        unpacked_int8 = unpack_int4(self.weight_packed)
        return dequantize_symmetric(unpacked_int8, self.weight_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly INT4 weight unpacking and dequantization."""
        w_dequant = self.dequantize_weight()
        return F.linear(x, w_dequant, self.bias)

    def memory_bytes(self) -> int:
        """Calculates total memory occupied by packed weights and scaling parameters in bytes."""
        total = self.weight_packed.nelement() * 1  # 1 byte stores two 4-bit weights
        total += self.weight_scale.nelement() * 4
        if self.bias is not None:
            total += self.bias.nelement() * 4
        return total
