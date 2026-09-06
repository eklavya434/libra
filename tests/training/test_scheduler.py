"""
Unit tests for CosineWarmupScheduler.
"""

import math

from packages.training.scheduler import CosineWarmupScheduler


def test_cosine_warmup_scheduler_stages():
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_steps = 20
    max_steps = 100

    scheduler = CosineWarmupScheduler(
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )

    # 1. Warmup start: step 1 should be 1/20 of max_lr
    lr_1 = scheduler.get_lr(1)
    assert math.isclose(lr_1, max_lr * (1 / 20), rel_tol=1e-5)

    # 2. Peak at end of warmup: step 20 should equal max_lr
    lr_warmup = scheduler.get_lr(warmup_steps)
    assert math.isclose(lr_warmup, max_lr, rel_tol=1e-5)

    # 3. Midway through decay: step 60 (halfway between 20 and 100)
    lr_mid = scheduler.get_lr(60)
    expected_mid = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * 0.5))
    assert math.isclose(lr_mid, expected_mid, rel_tol=1e-5)

    # 4. Floor after max_steps: step 100 and step 150 should equal min_lr
    assert math.isclose(scheduler.get_lr(max_steps), min_lr, rel_tol=1e-5)
    assert math.isclose(scheduler.get_lr(150), min_lr, rel_tol=1e-5)


def test_scheduler_state_dict_roundtrip():
    s1 = CosineWarmupScheduler(max_lr=2e-3, min_lr=2e-4, warmup_steps=50, max_steps=200)
    state = s1.state_dict()

    s2 = CosineWarmupScheduler(max_lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=50)
    s2.load_state_dict(state)

    assert s2.max_lr == 2e-3
    assert s2.warmup_steps == 50
    assert s2.get_lr(25) == s1.get_lr(25)
