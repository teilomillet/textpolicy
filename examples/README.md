# TextPolicy Examples - Learn to Train Better Models

Progressive examples showing how to build reward systems for reinforcement learning training of language models. Each example builds toward the ultimate goal: training better models.

## Learning Path to Model Training

### Step 1: [01_hello_reward.py](01_hello_reward.py) 
**Foundation of RL Training** *(45 lines)*
```bash
uv run python examples/01_hello_reward.py
```
**Learn:** Reward functions that tell models what "good" text looks like
**Training Context:** Without rewards, RL cannot train - this is the foundation

### Step 2: [02_reward_decorator.py](02_reward_decorator.py)
**Building Modular Training Systems** *(62 lines)*
```bash
uv run python examples/02_reward_decorator.py
```
**Learn:** @reward decorator makes functions discoverable by training systems
**Training Context:** Real training combines multiple rewards - modularity is essential

### Step 3: [03_batch_processing.py](03_batch_processing.py)
**Training Performance at Scale** *(73 lines)*
```bash
uv run python examples/03_batch_processing.py
```
**Learn:** MLX batch processing for thousands of texts per training step
**Training Context:** Training requires evaluating ~1000s of completions efficiently

### Step 4: [04_multiple_rewards.py](04_multiple_rewards.py)
**Real-World Training Quality** *(94 lines)*
```bash
uv run python examples/04_multiple_rewards.py
```
**Learn:** Combining multiple objectives (helpfulness, safety, accuracy)
**Training Context:** Production models optimize multiple objectives simultaneously

### Step 5: [05_textpolicy_essence.py](05_textpolicy_essence.py)
**Ready for Production Training** *(75 lines)*
```bash
uv run python examples/05_textpolicy_essence.py
```
**Experience:** Complete training-ready system in 50 lines
**Training Context:** You now understand everything needed to train better models

### Step 6: [06_minimal_training.py](06_minimal_training.py)
**Complete Working RL Training** *(130 lines)*
```bash
uv run python examples/06_minimal_training.py
```
**Experience:** Actual RL training loop that improves model behavior
**Training Context:** See measurable learning improvement using your reward system

## Quick Start for Model Training

```bash
# Understand the foundation
uv run python examples/01_hello_reward.py

# Learn the training system
uv run python examples/02_reward_decorator.py

# See the complete system
uv run python examples/05_textpolicy_essence.py

# Watch actual RL training work!
uv run python examples/06_minimal_training.py
```

## Example Progression to Training

| Step | Lines | Concept | Training Relevance |
|------|-------|---------|-------------------|
| 01_hello_reward | 50 | Reward functions score text | Foundation - no rewards = no RL |
| 02_reward_decorator | 68 | @reward makes functions modular | Scale - real training needs many rewards |
| 03_batch_processing | 79 | MLX processes 1000s of texts | Performance - training requires speed |
| 04_multiple_rewards | 100 | Multi-objective optimization | Quality - real models need multiple goals |
| 05_textpolicy_essence | 75 | Complete training system | Ready - everything needed for RL training |
| 06_minimal_training | 130 | Working RL training loop | Proof - see actual model improvement |

**Total learning time:** 25 minutes from zero to working RL training

## Why This Progression Works

- **Clear goal:** Every step builds toward model training
- **Progressive complexity:** Each example adds exactly one concept
- **Training context:** Explains WHY each step matters for RL
- **Immediate feedback:** Run examples instantly, see results
- **Production ready:** After step 5, ready to train models

## Training Mastery

After these examples:
- Understand how RL training uses rewards
- Know how to create modular reward systems  
- Can process training-scale text batches efficiently
- Can balance multiple training objectives
- Have complete textpolicy system knowledge
- Have seen working RL training with measurable improvement

**Result: Complete mastery of RL training for language models**

## Requirements

```bash
uv add textpolicy
```

Each example is self-contained and explains its role in the training pipeline.

---

*Progressive learning path: Foundation -> Modularity -> Performance -> Quality -> Training Ready*