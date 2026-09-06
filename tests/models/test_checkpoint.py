"""
Unit test for checkpoint saving and loading roundtrip determinism.
"""

import os

import torch

from packages.models.config import TinyTransformerConfig
from packages.models.transformer import TinyTransformerLM
from packages.training.trainer import load_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    config = TinyTransformerConfig(
        vocab_size=32,
        max_context_length=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model = TinyTransformerLM(config)
    model.eval()

    test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        original_logits, _ = model(test_input)

    # Save checkpoint to temporary path
    ckpt_path = os.path.join(tmp_path, "test_ckpt.pt")
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "step": 10,
            "final_loss": 1.234,
        },
        ckpt_path,
    )

    # Reload model
    loaded_model, ckpt_data = load_checkpoint(ckpt_path)

    with torch.no_grad():
        loaded_logits, _ = loaded_model(test_input)

    assert ckpt_data["step"] == 10
    assert torch.allclose(original_logits, loaded_logits, atol=1e-6)
