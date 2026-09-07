"""
Libra Models - Core Quantization Mathematics from First Principles
Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)

Implements:
1. Symmetric Quantization (Signed INT8 / INT4, zero_point = 0)
2. Asymmetric Affine Quantization (Arbitrary distributions, non-zero zero_point)
3. INT4 Bit-Packing & Unpacking (2 nibbles packed into 1 uint8 byte)
4. Quantization Error Metrics (MSE, SQNR, Cosine Similarity)
"""

from __future__ import annotations

import math

import torch


def quantize_symmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a floating point tensor symmetrically into signed integers.

    Formula:
        q_max = 2**(bits - 1) - 1
        q_min = -2**(bits - 1)
        scale = max(|tensor|) / q_max
        tensor_q = clamp(round(tensor / scale), q_min, q_max)

    Args:
        tensor: Floating point tensor to quantize.
        bits: Bit width (typically 8 or 4).
        per_channel: If True, computes scale per slice along `dim`.
        dim: Dimension along which per-channel scaling is calculated.

    Returns:
        tuple of (quantized_tensor, scale)
    """
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))

    if per_channel:
        # Reduce over all dims except `dim`
        reduce_dims = [d for d in range(tensor.ndim) if d != dim]
        max_abs = tensor.abs()
        for d in sorted(reduce_dims, reverse=True):
            max_abs = max_abs.amax(dim=d, keepdim=True)
        scale = max_abs / q_max
    else:
        max_abs = tensor.abs().amax()
        scale = max_abs / q_max

    # Guard against division by zero for all-zero tensors
    scale = torch.clamp(scale, min=1e-8)

    quantized = torch.clamp(torch.round(tensor / scale), min=q_min, max=q_max)

    if bits == 8:
        quantized = quantized.to(torch.int8)
    else:
        quantized = quantized.to(torch.int32)

    return quantized, scale


def dequantize_symmetric(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes a symmetrically quantized integer tensor back to float32.

    Formula:
        tensor_rec = quantized.float() * scale
    """
    return quantized.to(torch.float32) * scale


def quantize_asymmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes a floating point tensor asymmetrically into unsigned integers [0, 2**bits - 1].

    Formula:
        q_max = 2**bits - 1
        scale = (max(tensor) - min(tensor)) / q_max
        zero_point = round(-min(tensor) / scale)
        tensor_q = clamp(round(tensor / scale) + zero_point, 0, q_max)

    Returns:
        tuple of (quantized_tensor, scale, zero_point)
    """
    q_max = (1 << bits) - 1

    if per_channel:
        reduce_dims = [d for d in range(tensor.ndim) if d != dim]
        min_val = tensor
        max_val = tensor
        for d in sorted(reduce_dims, reverse=True):
            min_val = min_val.amin(dim=d, keepdim=True)
            max_val = max_val.amax(dim=d, keepdim=True)
    else:
        min_val = tensor.amin()
        max_val = tensor.amax()

    scale = torch.clamp((max_val - min_val) / q_max, min=1e-8)
    zero_point = torch.round(-min_val / scale)

    quantized = torch.clamp(torch.round(tensor / scale) + zero_point, min=0, max=q_max)

    if bits == 8:
        quantized = quantized.to(torch.uint8)
    else:
        quantized = quantized.to(torch.int32)

    return quantized, scale, zero_point


def dequantize_asymmetric(
    quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantizes an asymmetric integer tensor back to float32.

    Formula:
        tensor_rec = (quantized.float() - zero_point.float()) * scale
    """
    return (quantized.to(torch.float32) - zero_point.to(torch.float32)) * scale


def pack_int4(int4_tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs two signed 4-bit values (range [-8, 7]) into one uint8 byte.
    The last dimension of `int4_tensor` must be even.

    Encoding:
        low_nibble = val1 & 0x0F
        high_nibble = val2 & 0x0F
        packed = (high_nibble << 4) | low_nibble
    """
    orig_shape = int4_tensor.shape
    if orig_shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even for INT4 packing, got {orig_shape[-1]}")

    # Shift signed [-8, 7] into unsigned [0, 15]
    unsigned_nibbles = (int4_tensor.to(torch.int32) & 0x0F).to(torch.uint8)

    low = unsigned_nibbles[..., 0::2]
    high = unsigned_nibbles[..., 1::2]

    packed = (high << 4) | low
    return packed


def unpack_int4(packed_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unpacks one uint8 byte back into two signed 4-bit integers [-8, 7].
    The resulting tensor will have double the size on the last dimension.
    """
    low = (packed_tensor & 0x0F).to(torch.int32)
    high = ((packed_tensor >> 4) & 0x0F).to(torch.int32)

    # Convert unsigned 4-bit [0, 15] back to signed 2's complement [-8, 7]
    # If 4th bit is 1 (val >= 8), subtract 16
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    # Interleave along the last dimension
    orig_shape = list(packed_tensor.shape)
    orig_shape[-1] = orig_shape[-1] * 2

    unpacked = torch.empty(orig_shape, dtype=torch.int8, device=packed_tensor.device)
    unpacked[..., 0::2] = low.to(torch.int8)
    unpacked[..., 1::2] = high.to(torch.int8)

    return unpacked


def compute_quantization_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> dict[str, float]:
    """
    Computes fidelity metrics between original FP32 tensor and reconstructed tensor:
    - Mean Squared Error (MSE)
    - Signal-to-Quantization-Noise Ratio (SQNR in dB)
    - Cosine Similarity
    """
    orig = original.to(torch.float32)
    rec = reconstructed.to(torch.float32)

    mse = torch.mean((orig - rec) ** 2).item()
    signal_power = torch.mean(orig**2).item()

    if mse <= 1e-12:
        sqnr_db = 100.0  # Cap perfect reconstruction
    elif signal_power <= 1e-12:
        sqnr_db = 0.0
    else:
        sqnr_db = 10.0 * math.log10(signal_power / max(mse, 1e-12))

    orig_flat = orig.reshape(1, -1)
    rec_flat = rec.reshape(1, -1)
    cos_sim = torch.nn.functional.cosine_similarity(orig_flat, rec_flat).item()

    return {
        "mse": round(mse, 6),
        "sqnr_db": round(sqnr_db, 2),
        "cosine_similarity": round(cos_sim, 6),
    }
