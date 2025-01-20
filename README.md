# Feasible Learning

The `trl` directory contains the code to run Orca experiments. This code is primarily built on Hugging Face's TRL library, which is why it is maintained separately from the main codebase. For additional details, please refer to the README file located within the `trl` directory.

## Usage

**CIFAR-10**

```bash  
python main.py \
    --data_config=configs/data.py:"dataset_type=cifar10" \
    --model_config=configs/model.py:"model_type=CIFAR_ResNet18_SGD model.init_seed=135" \
    --task_config=configs/task.py:"task_type=feasible_classification task.pointwise_probability=0.6 optim.dual_optimizer.kwargs.lr=1e-4" \
    --config.train.total_epochs=200 --config.train.seed=0 \
    --metrics_config=configs/metrics.py:"classification" \
    --resources_config=configs/resources.py:"job_type=debug" \
    --config.logging.wandb_mode="online"  --config.logging.wandb_tags="('debug','aistats')" \
    -config.logging.save_losses_and_multipliers=True --config.logging.save_eval_logs_every_n_epochs=1
```

**UTKFace**

```bash
python main.py \
    --data_config=configs/data.py:"dataset_type=utkface" \
    --model_config=configs/model.py:"model_type=UTKFace_ResNet18" \
    --task_config=configs/task.py:"task_type=feasible_regression task.pointwise_loss=0.02 optim.dual_optimizer.kwargs.lr=1e-3" \
    --config.train.total_epochs=150 --config.train.seed=0 \
    --metrics_config=configs/metrics.py:"regression" \
    --resources_config=configs/resources.py:"job_type=debug" \
    --config.logging.wandb_mode="online"  --config.logging.wandb_tags="('debug','aistats')" \
    -config.logging.save_losses_and_multipliers=True --config.logging.save_eval_logs_every_n_epochs=1
```

**Two-Moons**

```bash
python main.py \
    --data_config=configs/data.py:"dataset_type=two_moons" \
    --model_config=configs/model.py:"model_type=TwoDim_MLP optim.primal_optimizer.kwargs.lr=5e-4" \
    --task_config=configs/task.py:"task_type=feasible_classification task.pointwise_probability=0.6 optim.dual_optimizer.kwargs.lr=1e-2" \
    --config.train.total_epochs=250 --config.train.seed=0 \
    --metrics_config=configs/metrics.py:"classification" \
    --resources_config=configs/resources.py:"job_type=debug" \
    --config.logging.wandb_mode="online"  --config.logging.wandb_tags="('debug','aistats')" \
    -config.logging.save_losses_and_multipliers=True --config.logging.save_eval_logs_every_n_epochs=1
```


Note that the parsed text arguments for file-based configs need to point to the "full
path" of the configuration item. For example, if you want to change `restart_on_feasible`
in `config.task.multiplier_kwargs`, you need to use
`task.multiplier_kwargs.restart_on_feasible=BOOL` (and NOT
`multiplier_kwargs.restart_on_feasible=BOOL`).

| Cluster | Is local? | Is SLURM? | Is interactive? | Is background? | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `debug` | ✅ | 🚫 | ✅ | 🚫 | For local execution on your machine or on a compute node |
| `local` | ✅ | 🚫 | 🚫 | ✅ | This could be locally on your machine or on a compute node |
| `main` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `main` partition |
| `long` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `long` partition |

- Keep in mind that you can run the code from a compute node on SLURM via `debug` or `local`.
- Currently, using multiple GPUs is not supported.

## Required enviroment variables

We use [`dotenv`](https://github.com/theskumar/python-dotenv) to manage environment variables. Please create a `.env` file in the root directory of the project and add the following variables:

```
# Location of the directory containing your datasets
DATA_DIR=

# The directory where the results will be saved
CHECKPOINT_DIR=

# If you want to use Weights & Biases, add the entity name here
WANDB_ENTITY=

WANDB_PROJECT=
# Directory for Weights & Biases local storage
WANDB_DIR=

# Directory for logs created by submitit
SUBMITIT_DIR=

KAGGLE_USERNAME=
KAGGLE_KEY=
```

## Code style

- This project uses `black`, `isort`, and `flake8` for enforcing code style. See `requirements.txt` for version numbers.
- We use `pre-commit` hooks to ensure that all code committed respects the code style.
- After (1) cloning the repo, (2) creating your environment and (3) installing the required
packages, you are strongly encouraged to run `pre-commit install` to set-up pre-commit hooks.

### Logging format

Whenever you are using logging inside a module, please remember to use the _rich_ formatting.

Do NOT do this:
```
import logging
logger = logging.getLogger(__name__)
```

DO this instead:
```
import shared
logger = shared.fetch_main_logger()
```
