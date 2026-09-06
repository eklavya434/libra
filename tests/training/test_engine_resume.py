"""
Unit test for TrainingEngine checkpoint save and resume functionality.
"""

import os

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.dataset import TextDataset
from packages.training.engine import TrainingConfig, TrainingEngine


def test_training_engine_pause_and_resume(tmp_path):
    config = ModernTransformerConfig(
        vocab_size=64,
        max_context_length=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
    )
    model = ModernTransformerLM(config)

    corpus = "The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step."
    dataset = TextDataset(corpus, train_ratio=0.8)

    ckpt_dir = str(tmp_path)
    train_cfg = TrainingConfig(
        max_steps=20,
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        eval_interval=10,
        eval_batches=2,
        checkpoint_dir=ckpt_dir,
    )

    # Phase 1: Train for 10 steps and save
    engine1 = TrainingEngine(model, dataset, train_cfg)
    train_cfg.max_steps = 10
    engine1.train(start_step=1)

    ckpt_path = os.path.join(ckpt_dir, "test_resume.pt")
    engine1.save_checkpoint(step=10, path=ckpt_path)
    assert os.path.exists(ckpt_path)

    # Phase 2: Create brand new model and engine, resume and continue to step 20
    model2 = ModernTransformerLM(config)
    train_cfg.max_steps = 20
    engine2 = TrainingEngine(model2, dataset, train_cfg)

    resumed_step = engine2.resume_from_checkpoint(ckpt_path)
    assert resumed_step == 11

    history = engine2.train(start_step=resumed_step)

    # Final step reached must be 20
    assert engine2.current_step == 20
    assert len(history) >= 1
