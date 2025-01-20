import logging
import re
from typing import Optional

import ml_collections as mlc
import rich.logging
import rich.text

MAIN_LOGGER_NAME = "TRAINER"

REGEX_PATTERN = r"((?:\w+\.)*\w+)=([-]?\d+(?:\.\d+)?(?:e[-+]?\d+)?|\w+|\([^)]*\))"


def fetch_main_logger(apply_basic_config=False):
    logger = logging.getLogger(MAIN_LOGGER_NAME)
    if apply_basic_config:
        configure_logger(logger)
    return logger


def configure_logger(logger, custom_format=None, level=logging.INFO, propagate=False, show_path=False):
    logger.propagate = propagate

    for handler in logger.handlers:
        logger.removeHandler(handler)

    format = f"%(module)s:%(funcName)s:%(lineno)d | %(message)s" if custom_format is None else custom_format
    log_formatter = logging.Formatter(format)

    rich_handler = rich.logging.RichHandler(
        markup=True,
        rich_tracebacks=True,
        omit_repeated_times=True,
        show_path=False,
        log_time_format=lambda dt: rich.text.Text.from_markup(f"[red]{dt.strftime('%y-%m-%d %H:%M:%S.%f')[:-4]}"),
    )
    rich_handler.setFormatter(log_formatter)
    logger.addHandler(rich_handler)

    logger.setLevel(level)


def drill_to_key_and_set_(working_dict, key, value) -> None:
    # Need to split the key by "." and traverse the config to set the new value
    split_key = key.split(".")
    entry_in_config = working_dict
    for subkey in split_key[:-1]:
        try:
            entry_in_config = entry_in_config[subkey]
        except:
            pass
    entry_in_config[split_key[-1]] = value


def update_config_with_cli_args_(config, variables):
    for key, value in variables.items():
        try:
            value = eval(value)
        except NameError:
            pass
        drill_to_key_and_set_(config, key=key, value=value)


def default_get_config(
    config_group_name: Optional[str], pop_key: str, preset_configs: dict, cli_cmd: Optional[str] = ""
) -> dict:
    """Wrapper for post-processing the output of a call to `get_config` triggered by
    mlCollections on the different configs.

    Args:
        config_group_name (Optional[str]): The name of the config group to return. For
            example, this can be `task`, `resources`, `model`, etc. When it is None,
            we assume that the `get_config` method returns a dictionary with sub-configs.
            For example, the `get_config` call on `configs/task.py` returns a dictionary
            with keys `task` and `optim`.
        pop_key (str): Key to pop from the CLI args containing the name of the chosen
            preset config.
        preset_configs (dict): A dictionary mapping the name of known preset configs to
            callables returning pre-populated configs.
        cli_args (Optional[str]): A string containing key-value pairs to update the
            preset config. The format is:
            "{pop_key}={CHOSEN_PRESET_CONFIG_NAME} key1=value1 key2=value2 ...".
    """

    # Extract the key-value pairs from the config string which has the format
    # "key1=value1 key2=value2 ..."
    cli_cmd = cli_cmd.strip() if cli_cmd is not None else ""
    matches = re.findall(REGEX_PATTERN, cli_cmd)

    cli_args = {key: value for key, value in matches}
    chosen_preset_config_name = cli_args.pop(pop_key)

    # Generate default preset config and update values according to CLI args
    preset_config = preset_configs[chosen_preset_config_name]()
    update_config_with_cli_args_(preset_config, cli_args)

    if isinstance(preset_config, dict):
        # If the preset config is a dictionary (not an mlc.ConfigDict!), we have a
        # multi-group config (like in `task` or `model`).
        assert config_group_name is None
        return preset_config
    elif isinstance(preset_config, mlc.ConfigDict):
        assert config_group_name is not None
        return {config_group_name: preset_config}
    else:
        raise RuntimeError(f"Preset config type for group {config_group_name} has unkonwn type {type(preset_config)}")
