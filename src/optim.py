from types import SimpleNamespace

import cooper
import torch

import shared

logger = shared.fetch_main_logger()


def build_cooper_optimizer_and_schedulers(model, cmp, config) -> tuple[cooper.optim.CooperOptimizer, SimpleNamespace]:
    primal_optimizer, primal_scheduler = build_optimizer_from_config(
        optimizer_config=config.optim.primal_optimizer, params=model.parameters()
    )

    total_params_in_optimizer = count_parameters_in_optimizer(primal_optimizer)
    num_params = sum([param.numel() for param in model.parameters()])
    logger.info(f"Created primal optimizer accounts for {total_params_in_optimizer}/{num_params} parameters")
    assert total_params_in_optimizer == num_params

    cooper_optimizer_kwargs = {"cmp": cmp, "primal_optimizers": primal_optimizer}

    if cmp.has_dual_variables:

        with config.optim.dual_optimizer.unlocked():
            # As dual variables aim to *maximize* the Lagrangian, we hard-code `maximize=True`
            config.optim.dual_optimizer.kwargs["maximize"] = True

            if config.optim.dual_optimizer.optimizer_class == torch.optim.SGD:
                # Pytorch 2.0 does not support `foreach` computation for SGD on parameters
                # with sparse gradients. Hard-coding `foreach=False` as SGD would produce
                # a an error otherwise: "Could not run 'aten::_foreach_neg' with arguments
                # from the 'SparseCUDA' backend".
                # See: https://github.com/pytorch/pytorch/blob/18f203a5678f1d29c4f3f8eecfee95f2206ad5ae/torch/optim/sgd.py#L326
                config.optim.dual_optimizer.kwargs["foreach"] = False

        dual_optimizer, dual_scheduler = build_optimizer_from_config(config.optim.dual_optimizer, cmp.dual_parameters())
        cooper_optimizer_kwargs["dual_optimizers"] = dual_optimizer

        total_params_in_optimizer = count_parameters_in_optimizer(dual_optimizer)
        num_params = sum([param.numel() for param in cmp.dual_parameters()])
        logger.info(f"Created dual optimizer accounts for {total_params_in_optimizer}/{num_params} parameters")
        assert total_params_in_optimizer == num_params

    else:
        dual_scheduler = None

    schedulers_namespace = SimpleNamespace(primal=primal_scheduler, dual=dual_scheduler)

    cooper_optimizer = config.optim.cooper_optimizer_class(**cooper_optimizer_kwargs)
    logger.info(f"Created Cooper optimizer of type {type(cooper_optimizer)}")

    return cooper_optimizer, schedulers_namespace


def build_optimizer_from_config(optimizer_config, params):

    optimizer = optimizer_config.optimizer_class(params, **optimizer_config.kwargs)
    logger.info(f"Instantiated optimizer: \n {optimizer}")

    if optimizer_config.scheduler.scheduler_class is not None:
        scheduler = optimizer_config.scheduler.scheduler_class(optimizer, **optimizer_config.scheduler.kwargs)
        logger.info(f"Created {optimizer_config.scheduler.scheduler_class} scheduler")
    else:
        scheduler = None

    return optimizer, scheduler


def count_parameters_in_optimizer(optimizer):
    total_params_in_optimizer = 0
    for param_group in optimizer.param_groups:
        total_params_in_optimizer += sum([param.numel() for param in param_group["params"]])
    return total_params_in_optimizer
