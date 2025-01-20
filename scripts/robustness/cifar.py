import copy
import itertools
import subprocess
from typing import Literal

# Set IS_DRAFT to True to print the command without running it
IS_DRAFT = False

# --------------------------------------------------------------------------------------
WANDB_MODE = "online"
WANDB_TAGS = "('robustness','aistats')"

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
    -config.logging.save_losses_and_multipliers=False --config.logging.save_eval_logs_every_n_epochs=2000
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
POINTWISE_PROBABILITIES = [1.0]
PRIMAL_LRS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
DUAL_LRS = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

for seed in TRAIN_SEEDS:
    for primal_lr in PRIMAL_LRS:

        SHARED_KWARGS["MODEL_CONFIG"] = f"{MODEL_CONFIG} optim.primal_optimizer.kwargs.lr={primal_lr}"

        ## ERM
        TASK_CONFIG = "task_type=erm_classification"
        cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
        run_command(cmd)

        # --------------------------------------------------------------------------------------
        for pw_prob in POINTWISE_PROBABILITIES:

            ## CSERM
            TASK_CONFIG = f"task_type=cserm_classification task.pointwise_probability={pw_prob}"
            cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
            run_command(cmd)

            for dual_lr in DUAL_LRS:

                ## FL
                TASK_CONFIG = f"task_type=feasible_classification task.pointwise_probability={pw_prob} task.early_stop_on_feasible=False"
                DUAL_CONFIG = generate_dual_optimizer_cmd(lr=dual_lr, weight_decay=0.0)
                TASK_CONFIG = f"{TASK_CONFIG} {DUAL_CONFIG}"
                cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
                run_command(cmd)

                ## RFL
                TASK_CONFIG = f"task_type=feasible_classification task.pointwise_probability={pw_prob} task.early_stop_on_feasible=False"
                DUAL_CONFIG = generate_dual_optimizer_cmd(
                    lr=dual_lr, weight_decay=1.0
                )  # RFL is enabled by setting non-zero weight decay
                TASK_CONFIG = f"{TASK_CONFIG} {DUAL_CONFIG}"
                cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
                run_command(cmd)
