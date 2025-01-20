
# Code for LLm experiments based on TRL - Transformer Reinforcement Learning

This is an implementation of feasible learning for language models (LLMs) based on the TRL library by Hugging Face. 

The implementation of FL, RFL and clamped-erm for the DPO loss can be found un `trl/trainer/dpo_trainer_feas.py`.

The original ERM dpo implementation can be found in `examples/scripts/dpo.py`.

The experiments reported in the paper for the orca dataset can be found in `examples/scripts/paper_experiments/`.


### Installation

Clone the repository and install it with pip:
```bash
pip install .
```

## How to use

### DPOfTrainer

The `DPOfTrainer` is a subclass of `Trainer` that implements the DPO loss for feasible learning. It can be used as a drop-in replacement for the `DPOTrainer` class.

It has the following additional parameters:

- **dual_lr**: `float = field(default=1.0, metadata={"help": "Dual Learning Rate"})`
- **resilient_alpha**: `float = field(default=2.0, metadata={"help": "Dual Weight Decay Coefficient"})`
- **loss_tolerance**: `float = field(default=1e-3, metadata={"help": "Loss Tolerance"})`
- **algorithm**: `str = field(default="erm", metadata={"help": "Algorithm can be 'erm', 'clamped' or 'feasible'"})`