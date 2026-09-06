# Educational Guide: Phase 5 — Training Engine

Welcome to **Phase 5** of **Libra**!

In this phase, we constructed a **reusable, production-grade training engine** equipped with the core optimization and stabilization algorithms used in real-world LLM pre-training.

---

## 1. WHAT Was Built?

1. **Cosine Warmup Scheduler** (`packages/training/scheduler.py`): Two-phase learning rate schedule: linear warmup followed by half-cosine annealing.
2. **Metrics & Telemetry** (`packages/training/metrics.py`): Real-time computation of **Perplexity** ($\text{PPL} = e^{\mathcal{L}}$), token throughput (tokens/second), and elapsed time.
3. **Reusable Training Engine** (`packages/training/engine.py`):
   - **Gradient Accumulation**: Simulating larger effective batch sizes without exceeding our 16 GB RAM ceiling.
   - **Gradient Norm Clipping**: Shielding against numerical overflow and exploding gradients.
   - **Validation & Checkpointing**: Periodic evaluation on unseen holdout data.
   - **Fault-Tolerant Resumption**: Saving complete optimizer, scheduler, and model states so training can be paused and resumed seamlessly.
4. **Declarative Training Config** (`configs/training/cpu_quick_train.yaml`): Easy hyperparameter tuning via YAML.

---

## 2. WHY Do We Need Each Technique?

### A. Gradient Accumulation (Simulating Big Hardware on a CPU)
In deep learning, small batch sizes (e.g. 1 or 2 sequences) produce noisy, erratic gradient vectors that bounce around the loss landscape. Larger batch sizes produce smooth, accurate gradient steps.
* **The Problem**: A batch size of 64 on our laptop would consume too much memory and cause an Out-Of-Memory (OOM) error.
* **The Solution**: We process 2 smaller micro-batches of size 4, calculate gradients for each, add (accumulate) them together, and only update the model weights once! We get the exact mathematical benefits of batch size 8 while only storing batch size 4 in memory.

### B. Learning Rate Warmup & Cosine Decay
Why can't we just use a constant learning rate?
1. **Early "Gradient Shock"**: At step 0, all weights are randomly initialized. Gradients are huge and erratic. If the learning rate is high, the model takes wild leaps and diverges immediately. **Warmup** starts $lr$ near 0 and gently increases it over the first 25–100 steps while the weights stabilize.
2. **Late "Settling"**: Once the model has learned general sentence structure, large learning rate steps will bounce over the narrow global minimum. **Cosine decay** gradually cools the learning rate down, letting the model settle into the sharpest, most accurate minimum.

### C. What is Perplexity ($\text{PPL}$)?
Perplexity is the gold-standard metric for evaluating language models:
$$\text{PPL} = e^{\text{Cross-Entropy Loss}}$$

**Intuitive Explanation**:
Perplexity represents the **"uncertainty branching factor"**:
- If a model has a vocabulary of 512 tokens and guesses completely randomly:
  $$\text{Loss} = -\ln(1/512) \approx 6.24 \implies \text{PPL} = e^{6.24} = 512.0$$
  The model is as confused as picking out of 512 hats.
- After training, when loss drops to $0.41$:
  $$\text{PPL} = e^{0.41} \approx 1.51$$
  The model is now so confident that, on average, it is choosing between only **1.5 likely possibilities**!

---

## 3. HOW Does Checkpoint Resumption Work?

When a training run is paused, saving only the model weights (`model.state_dict()`) is **not enough**:
- Optimizers like AdamW maintain running averages of past gradients (momentum $m_t$ and variance $v_t$).
- If you resume training without restoring the optimizer's momentum, the optimizer will experience a shock, and the loss will temporarily spike.

Our `TrainingEngine` serializes:
1. `model_state_dict`: Current neural network weights.
2. `optimizer_state_dict`: AdamW momentum and variance buffers.
3. `scheduler_state_dict`: Current learning rate position along the cosine wave.
4. `step`: Exact step count to resume from ($K + 1$).

---

## 4. How We Verified It

We ran automated tests in `tests/training/`:
1. `test_scheduler.py`: Verified linear warmup slope and cosine decay endpoints.
2. `test_gradient_accumulation.py`: Mathematically proved that accumulating 2 micro-steps produces identical gradients to a single combined batch.
3. `test_engine_resume.py`: Verified pausing at step 10, saving, and resuming cleanly to step 20 with zero state leakage.

---

## 5. NEXT: What Comes Next?

In **Phase 6: Evaluation**, we will build a comprehensive evaluation harness:
- Systematic validation loss and perplexity benchmarking.
- Generation quality tests across reasoning, mathematics, coding, and instruction following.
- Preparation for the **Comprehension Gate** before Phase 7!