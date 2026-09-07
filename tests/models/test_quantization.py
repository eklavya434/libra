"""
Tests for Core Quantization, Quantized Layers, and Post-Training Quantization
Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)
"""

import torch
from torch import nn

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.quantization.ptq import (
    audit_quantization_fidelity,
    quantize_model,
)
from packages.models.quantization.quant_core import (
    compute_quantization_metrics,
    dequantize_asymmetric,
    dequantize_symmetric,
    pack_int4,
    quantize_asymmetric,
    quantize_symmetric,
    unpack_int4,
)
from packages.models.quantization.quant_linear import (
    QuantizedLinearINT4,
    QuantizedLinearINT8,
)


def test_symmetric_quantization_int8():
    torch.manual_seed(42)
    x = torch.randn(4, 64, dtype=torch.float32)

    # Per-tensor
    x_q, scale = quantize_symmetric(x, bits=8, per_channel=False)
    assert x_q.dtype == torch.int8
    assert x_q.shape == x.shape
    assert scale.numel() == 1

    x_rec = dequantize_symmetric(x_q, scale)
    metrics = compute_quantization_metrics(x, x_rec)
    assert metrics["cosine_similarity"] > 0.999
    assert metrics["sqnr_db"] > 40.0  # INT8 typically > 40-50 dB SQNR


def test_symmetric_quantization_per_channel():
    torch.manual_seed(42)
    x = torch.randn(8, 32, dtype=torch.float32)

    x_q, scale = quantize_symmetric(x, bits=8, per_channel=True, dim=0)
    assert scale.shape == (8, 1)

    x_rec = dequantize_symmetric(x_q, scale)
    metrics = compute_quantization_metrics(x, x_rec)
    assert metrics["cosine_similarity"] > 0.999


def test_asymmetric_quantization():
    torch.manual_seed(42)
    # Strongly skewed non-zero distribution
    x = torch.rand(4, 32, dtype=torch.float32) * 5.0 + 2.0

    x_q, scale, zp = quantize_asymmetric(x, bits=8)
    assert x_q.dtype == torch.uint8
    assert (x_q >= 0).all() and (x_q <= 255).all()

    x_rec = dequantize_asymmetric(x_q, scale, zp)
    metrics = compute_quantization_metrics(x, x_rec)
    assert metrics["cosine_similarity"] > 0.999
    assert metrics["mse"] < 0.01


def test_int4_pack_and_unpack():
    # Valid signed 4-bit range: [-8, 7]
    int4_vals = torch.tensor([[-8, 7, 0, -1, 3, -4], [5, -5, 2, -2, 1, -1]], dtype=torch.int8)
    packed = pack_int4(int4_vals)
    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 3)  # Half of 6 columns

    unpacked = unpack_int4(packed)
    assert unpacked.dtype == torch.int8
    assert unpacked.shape == int4_vals.shape
    assert torch.equal(int4_vals, unpacked)


def test_quantized_linear_int8_forward():
    torch.manual_seed(42)
    linear = nn.Linear(32, 64, bias=True)
    qlinear = QuantizedLinearINT8.from_float(linear, per_channel=True)

    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        out_float = linear(x)
        out_quant = qlinear(x)

    assert out_quant.shape == (2, 5, 64)
    metrics = compute_quantization_metrics(out_float, out_quant)
    assert metrics["cosine_similarity"] > 0.995

    # Memory reduction
    float_bytes = linear.weight.nelement() * 4 + linear.bias.nelement() * 4
    quant_bytes = qlinear.memory_bytes()
    assert quant_bytes < float_bytes * 0.35  # Close to ~4x reduction


def test_quantized_linear_int4_forward():
    torch.manual_seed(42)
    linear = nn.Linear(32, 64, bias=True)
    qlinear = QuantizedLinearINT4.from_float(linear, per_channel=True)

    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        out_float = linear(x)
        out_quant = qlinear(x)

    assert out_quant.shape == (2, 5, 64)
    metrics = compute_quantization_metrics(out_float, out_quant)
    assert metrics["cosine_similarity"] > 0.97

    # Memory reduction: INT4 packed has 8x weight reduction
    float_weight_bytes = linear.weight.nelement() * 4
    packed_bytes = qlinear.weight_packed.nelement()
    assert packed_bytes == float_weight_bytes // 8


def test_post_training_quantize_model():
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(vocab_size=50, d_model=32, n_heads=4, n_layers=2)
    model = ModernTransformerLM(cfg)
    model.eval()

    sample_input = torch.randint(0, 50, (1, 8))

    # INT8 Quantization
    q_model_8 = quantize_model(model, mode="int8")
    assert any(isinstance(m, QuantizedLinearINT8) for m in q_model_8.modules())

    audit_8 = audit_quantization_fidelity(model, q_model_8, sample_input)
    assert audit_8["fidelity"]["cosine_similarity"] > 0.98

    # INT4 Quantization
    q_model_4 = quantize_model(model, mode="int4")
    assert any(isinstance(m, QuantizedLinearINT4) for m in q_model_4.modules())

    audit_4 = audit_quantization_fidelity(model, q_model_4, sample_input)
    assert audit_4["fidelity"]["cosine_similarity"] > 0.90
