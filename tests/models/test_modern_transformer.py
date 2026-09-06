"""
Unit tests for ModernTransformerLM and YAML configuration loading.
"""

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


def test_modern_transformer_from_yaml():
    config = ModernTransformerConfig.from_yaml("configs/models/tiny_modern_llama.yaml")
    assert config.vocab_size == 512
    assert config.d_model == 128
    assert config.tie_weights is False

    model = ModernTransformerLM(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits, loss = model(x, targets)

    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0


def test_modern_transformer_weight_tying():
    config_untied = ModernTransformerConfig.from_yaml("configs/models/tiny_modern_llama.yaml")
    config_tied = ModernTransformerConfig.from_yaml("configs/models/tiny_modern_tied.yaml")

    model_untied = ModernTransformerLM(config_untied)
    model_tied = ModernTransformerLM(config_tied)

    # Untied model has separate weights for embedding and output head
    assert model_untied.output_head.weight is not model_untied.tok_emb.weight

    # Tied model shares the exact same tensor parameter
    assert model_tied.output_head.weight is model_tied.tok_emb.weight

    # Tied model parameter count must be exactly vocab_size * d_model smaller
    param_diff = model_untied.count_parameters() - model_tied.count_parameters()
    expected_diff = config_tied.vocab_size * config_tied.d_model
    assert param_diff == expected_diff
