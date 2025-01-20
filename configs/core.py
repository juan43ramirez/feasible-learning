import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def get_config():
    config = mlc.ConfigDict()

    config.exp_name = MLC_PH(str)
    config.allow_silent_failure = False

    config.train = mlc.ConfigDict()
    config.train.seed = 0
    config.train.use_deterministic_ops = True
    config.train.total_epochs = MLC_PH(int)
    config.train.total_steps = MLC_PH(int)

    # These configs rely on separate config_files. See header of `main.py`
    config.data = mlc.ConfigDict()
    config.model = mlc.ConfigDict()
    config.task = mlc.ConfigDict()
    config.metrics = mlc.ConfigDict()
    config.resources = mlc.ConfigDict()

    # These configs are usually set in the `model` or `task` configs
    config.optim = mlc.ConfigDict()
    config.optim.cooper_optimizer_name = MLC_PH(str)
    config.optim.primal_optimizer = mlc.ConfigDict()
    config.optim.dual_optimizer = mlc.ConfigDict()

    # Fixed defaults for logging across all experiments
    config.logging = mlc.ConfigDict()
    config.logging.log_level = "INFO"
    config.logging.print_train_stats_period_steps = 100
    config.logging.eval_period_epochs = 1
    config.logging.eval_period_steps = MLC_PH(int)
    config.logging.save_eval_logs_every_n_epochs = 3
    config.logging.wandb_mode = "disabled"
    config.logging.wandb_tags = MLC_PH(tuple)
    config.logging.save_losses_and_multipliers = False
    config.logging.run_name = MLC_PH(str)

    config.checkpointing = mlc.ConfigDict()
    config.checkpointing.enabled = True
    # This `resume_from_checkpoint` entry is modified by __submitit_checkpoint__ upon preemption
    config.checkpointing.resume_from_checkpoint = False

    return config
