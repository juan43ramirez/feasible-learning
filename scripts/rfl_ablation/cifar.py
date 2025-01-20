import copy
import itertools
import subprocess
from typing import Literal

# Set IS_DRAFT to True to print the command without running it
IS_DRAFT = False

# --------------------------------------------------------------------------------------
WANDB_MODE = "online"
WANDB_TAGS = "('ablation','aistats')"

DATA_CONFIG = "dataset_type=cifar10"
MODEL_CONFIG = "model_type=CIFAR_ResNet18_SGD model.init_seed=135"
EPOCHS = 200

METRICS_CONFIG = "classification"

SHARED_KWARGS = dict(
    DATA_CONFIG=DATA_CONFIG,
    MODEL_CONFIG=MODEL_CONFIG,
    EPOCHS=EPOCHS,
    METRICS_CONFIG=METRICS_CONFIG,
    WANDB_MODE=WANDB_MODE,
    WANDB_TAGS=WANDB_TAGS,
)
# --------------------------------------------------------------------------------------


def run_command(cmd, is_draft=IS_DRAFT):
    print(cmd) if is_draft else subprocess.run(cmd, shell=True)


cmd_template = f"""python main.py \
    --data_config=configs/data.py:"{{DATA_CONFIG}}" \
    --model_config=configs/model.py:"{{MODEL_CONFIG}}" \
    --task_config=configs/task.py:"{{TASK_CONFIG}}" \
    --config.train.total_epochs={{EPOCHS}} --config.train.seed={{TRAIN_SEED}}\
    --metrics_config=configs/metrics.py:"{{METRICS_CONFIG}}" \
    --resources_config=configs/resources.py:"{{RESOURCES}}" \
    --config.logging.wandb_mode="{{WANDB_MODE}}"  --config.logging.wandb_tags="{{WANDB_TAGS}}" \
    -config.logging.save_losses_and_multipliers=True --config.logging.save_eval_logs_every_n_epochs=1
    """


def generate_resources_cmd(
    job_type: Literal["debug", "local", "unkillable", "main", "long", "long_cpu", "long_with_RTX"],
    num_gpus: int = 1,
    timeout_min: int = 60,
):
    return f"job_type={job_type} tasks_per_node={num_gpus} timeout_min={timeout_min}"


def generate_dual_optimizer_cmd(lr, weight_decay=0.0):
    return f"optim.dual_optimizer.kwargs.lr={lr} optim.dual_optimizer.kwargs.weight_decay={weight_decay}"


################################# Default Configs ######################################
RESOURCES = generate_resources_cmd(job_type="long", num_gpus=1, timeout_min=200)
SHARED_KWARGS["RESOURCES"] = RESOURCES

#######################################################################################

TRAIN_SEEDS = [0, 1, 2, 3, 4]
POINTWISE_PROBABILITIES = [0.6, 1.0]
dual_lr = 1e-4

WEIGHT_DECAYS = [0.0, 1.0, 10.0, 100.0, 1000.0]

for seed in TRAIN_SEEDS:
    for pw_prob in POINTWISE_PROBABILITIES:
        for weight_decay in WEIGHT_DECAYS:

            ## RFL
            TASK_CONFIG = f"task_type=feasible_classification task.pointwise_probability={pw_prob}"
            DUAL_CONFIG = generate_dual_optimizer_cmd(lr=dual_lr, weight_decay=weight_decay)
            TASK_CONFIG = f"{TASK_CONFIG} {DUAL_CONFIG}"
            cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
            run_command(cmd)
