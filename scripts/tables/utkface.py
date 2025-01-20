import copy
import itertools
import subprocess
from typing import Literal

# Set IS_DRAFT to True to print the command without running it
IS_DRAFT = False

# --------------------------------------------------------------------------------------
WANDB_MODE = "online"
WANDB_TAGS = "('utkface','aistats')"

DATA_CONFIG = "dataset_type=utkface"
MODEL_CONFIG = "model_type=UTKFace_ResNet18"
EPOCHS = 150

METRICS_CONFIG = "regression"

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
    -config.logging.save_losses_and_multipliers=True --config.logging.save_eval_logs_every_n_epochs=2000
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
RESOURCES = generate_resources_cmd(job_type="debug", num_gpus=1, timeout_min=60)
SHARED_KWARGS["RESOURCES"] = RESOURCES

#######################################################################################

TRAIN_SEEDS = [0, 1, 2, 3, 4]
POINTWISE_LOSSES = [0.02, 0.0]
cserm_lr = 1e-5
dual_lr = 1e-3

for seed in TRAIN_SEEDS:

    ## ERM
    TASK_CONFIG = "task_type=erm_regression"
    cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
    run_command(cmd)

    # --------------------------------------------------------------------------------------

    for pw_loss in POINTWISE_LOSSES:

        ## CSERM
        TASK_CONFIG = f"task_type=cserm_regression task.pointwise_loss={pw_loss}"
        CSERM_KWARGS = copy.copy(SHARED_KWARGS)
        # Needed to reduce the LR for this task since squared loss made diverge
        CSERM_KWARGS["MODEL_CONFIG"] = f"{MODEL_CONFIG} optim.primal_optimizer.kwargs.lr={cserm_lr}"
        cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **CSERM_KWARGS)
        run_command(cmd)

        ## FL
        TASK_CONFIG = f"task_type=feasible_regression task.pointwise_loss={pw_loss}"
        DUAL_CONFIG = generate_dual_optimizer_cmd(lr=dual_lr, weight_decay=0.0)
        TASK_CONFIG = f"{TASK_CONFIG} {DUAL_CONFIG}"
        cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
        run_command(cmd)

        ## RFL
        TASK_CONFIG = f"task_type=feasible_regression task.pointwise_loss={pw_loss}"
        DUAL_CONFIG = generate_dual_optimizer_cmd(
            lr=dual_lr, weight_decay=1.0
        )  # RFL is enabled by setting non-zero weight decay
        TASK_CONFIG = f"{TASK_CONFIG} {DUAL_CONFIG}"
        cmd = cmd_template.format(TASK_CONFIG=TASK_CONFIG, TRAIN_SEED=seed, **SHARED_KWARGS)
        run_command(cmd)
