#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
Validate model F0.5 performance on GEC dataset in post_validate()
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to set up root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
# from fairseq.trainer import Trainer
from trainer import Trainer
from fairseq.options import add_generation_args


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
            distributed_utils.is_master(cfg.distributed_training)
            and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    task.cfg_all = cfg

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    task.trainer = trainer
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
        cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        # valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
        # exit(0)

        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        valid_subsets: List[str],
        end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if 0 < cfg.optimization.stop_time_hours < training_time_hours:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
            (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
            or should_stop
            or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.checkpoint.save_interval_updates == 0
                    and num_updates >= cfg.dataset.validate_after_updates
            )
    )
    do_validate = (
            (
                    (not end_of_epoch and do_save)  # validate during mid-epoch saves
                    or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
                    or should_stop
                    or (
                            cfg.dataset.validate_interval_updates > 0
                            and num_updates > 0
                            and num_updates % cfg.dataset.validate_interval_updates == 0
                    )
            )
            and not cfg.dataset.disable_validation
            and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses , stats = [None], None
    if do_validate:
        valid_losses, stats = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    # Post Validate
    if do_validate and hasattr(task, "post_validate"):
        task.post_validate(trainer.get_model(), stats)

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        subsets: List[str],
) -> Tuple[List[Optional[float]], Dict[str, Any]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses , stats = [], None
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                        cfg.dataset.max_valid_steps is not None
                        and i > cfg.dataset.max_valid_steps
                ):
                    break
                trainer.valid_step(sample)

        # log validation stats
        # only tracking the best metric on the 1st validation subset
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)

        # Note by yejh: Delay `post_validate` until after `save_checkpoint`
        # if hasattr(task, "post_validate"):
        #     task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses, stats


def get_valid_stats(
        cfg: DictConfig,
        trainer: Trainer,
        stats: Dict[str, Any],
        tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    add_generation_args(parser)
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()

"""
============================== English Baseline ==============================
bart/preprocess/eng/real/bea_train2/bin
    --save-dir exps/temp
    --user-dir /data/yejh/nlp/GEC/models/backtranslation/
    --arch two_stage_bart_large
    --task translation
    --restore-file ../../../resources/bart.large/model.pt
    --reset-lr-scheduler
    --reset-optimizer
    --reset-meters
    --reset-dataloader
    --max-tokens 1024
    --optimizer adam
    --layernorm-embedding
    --weight-decay 0.01
    --share-all-embeddings
    --share-decoder-input-output-embed
    --update-freq 1
    --lr 3e-06
    -s src
    -t tgt
    --dropout 0.3
    --lr-scheduler inverse_sqrt
    --clip-norm 0.1
    --criterion label_smoothed_cross_entropy
    --label-smoothing 0.1
    --max-epoch 60
    --patience 10
    --adam-betas (0.9,0.999)
    --log-format tqdm
    --fp16
    --skip-invalid-size-inputs-valid-test
    --find-unused-parameters
    --keep-last-epochs 3
    
============================== English MixEdit ==============================
bart/preprocess/eng/real/bea_train2/bin
    --save-dir bart/exps/eng/temp
    --user-dir bart
    --arch bart_large
    --restore-file ../../../resources/bart.large/model.pt
    --task gec_dev
    --max-tokens 1024
    --optimizer adam
    --layernorm-embedding
    --weight-decay 0.01
    --share-all-embeddings
    --share-decoder-input-output-embed
    --update-freq 1
    --lr 3e-06
    --warmup-updates 100
    -s src
    -t tgt
    --dropout 0.3
    --lr-scheduler inverse_sqrt
    --clip-norm 0.1
    --criterion augmented_label_smoothed_cross_entropy
    --label-smoothing 0.1
    --max-epoch 2
    --patience 10
    --adam-betas (0.9,0.999)
    --log-format tqdm
    --fp16
    --skip-invalid-size-inputs-valid-test
    --find-unused-parameters
    --keep-last-epochs 1
    --reset-lr-scheduler
    --reset-optimizer
    --reset-meters
    --reset-dataloader
    --eval-gec
    --eval-gec-metric errant_eng
    --eval-gec-dataset wi_dev
    --eval-gec-output-prefix bart/exps/eng/temp/results/temp
    --beam 12
    --bpe gpt2
    --gpt2-encoder-json bart/preprocess/eng/encoder.json
    --gpt2-vocab-bpe bart/preprocess/eng/vocab.bpe
    --augmentation-schema trg_cut_off
    --augmentation-masking-probability 0.15
    --augmentation-masking-schema word
    --augmentation-enable-mix-edit
    --augmentation-pattern-noise-rate 0.15
    --file-dataset-m2 /vepfs/yejh/nlp/datasets/GEC/EGEC/bea19/WI/WI-train.errant
    --file-pattern bart/preprocess/eng/real/merge_pattern.json
    --regularization-weight 1.0
    
    
============================== English Transformer CutOff ==============================
transformer/preprocess/eng/real/bea_train1/bin
    --save-dir transformer/exps/eng/temp
    --user-dir bart
    --finetune-from-model transformer/exps/eng/cut_off/temp/checkpoint_last.pt
    --arch gec_transformer
    --task gec_dev
    --max-source-positions 512
    --max-target-positions 512
    --max-tokens 4096
    --optimizer adam
    --share-all-embeddings
    --share-decoder-input-output-embed
    --update-freq 4
    --lr 5e-4
    --warmup-init-lr 1e-07
    --warmup-updates 2000
    -s src
    -t tgt
    --lr-scheduler inverse_sqrt
    --clip-norm 0.1
    --criterion augmented_label_smoothed_cross_entropy
    --label-smoothing 0.1
    --max-epoch 300
    --patience 30
    --adam-betas (0.9,0.98)
    --log-format tqdm
    --fp16
    --find-unused-parameters
    --keep-last-epochs 1
    --eval-gec
    --eval-gec-min-update 0
    --eval-gec-metric errant_eng
    --eval-gec-dataset wi_dev
    --eval-gec-dataset-path transformer/preprocess/eng/bea_dev/valid.bpe.src
    --eval-gec-output-prefix transformer/exps/eng/temp/results/output
    --beam 12
    --remove-bpe
    --augmentation-schema cut_off
    --augmentation-masking-probability 0.15
    --augmentation-masking-schema word
    --regularization-weight 1.0
    --num-workers 1
    --seed 42
    
============================== Chinese Baseline ==============================
bart/preprocess/zho/real/mucgec_dev/bin
    --save-dir bart/exps/zho/temp
    --user-dir bart
    --finetune-from-model bart/exps/zho/real/chinese_hsk+lang8/checkpoint_best.pt
    --task gec_dev
    --arch gec_bart_large
    --max-tokens 4096
    --max-source-positions 1024
    --max-target-positions 1024
    --optimizer adam
    --layernorm-embedding
    --share-all-embeddings
    --share-decoder-input-output-embed
    --update-freq 4
    --lr 3e-05
    --warmup-updates 2000
    --weight-decay 0.01
    -s src
    -t tgt
    --dropout 0.2
    --lr-scheduler inverse_sqrt
    --clip-norm 1.0
    --criterion label_smoothed_cross_entropy
    --label-smoothing 0.1
    --max-epoch 100
    --patience 5
    --adam-betas (0.9,0.999)
    --log-format tqdm
    --fp16
    --skip-invalid-size-inputs-valid-test
    --find-unused-parameters
    --keep-last-epochs 5
    --remove-bpe
    --eval-gec
    --eval-gec-metric errant_zho
    --eval-gec-dataset mucgec_dev
    --eval-gec-output-prefix bart/exps/zho/temp/results/temp
    --beam 12
    
============================== Chinese MixEdit ==============================
bart/preprocess/zho/real/mucgec_dev/bin
    --save-dir bart/exps/zho/temp
    --user-dir bart
    --task gec_dev
    --arch gec_bart_large
    --restore-file transformers:fnlp/bart-large-chinese
    --max-tokens 2048
    --max-source-positions 1024
    --max-target-positions 1024
    --optimizer adam
    --layernorm-embedding
    --share-all-embeddings
    --share-decoder-input-output-embed
    --update-freq 4
    --lr 3e-05
    --warmup-updates 100
    --weight-decay 0.01
    -s src
    -t tgt
    --dropout 0.2
    --lr-scheduler inverse_sqrt
    --clip-norm 1.0
    --criterion augmented_label_smoothed_cross_entropy
    --label-smoothing 0.1
    --max-epoch 100
    --patience 5
    --adam-betas (0.9,0.999)
    --log-format tqdm
    --fp16
    --skip-invalid-size-inputs-valid-test
    --find-unused-parameters
    --keep-last-epochs 5
    --remove-bpe
    --eval-gec
    --eval-gec-metric errant_zho
    --eval-gec-dataset mucgec_dev
    --eval-gec-output-prefix bart/exps/zho/temp/results/temp
    --beam 12
    --augmentation-schema trg_cut_off
    --augmentation-masking-probability 0.20
    --augmentation-masking-schema word
    --augmentation-enable-mix-edit
    --mix-edit-remove-bpe ##
    --mix-edit-temperature 0.1
    --file-dataset-m2 bart/preprocess/zho/real/mucgec_dev/mucgec_dev.char.m2
    --file-pattern bart/preprocess/zho/real/hsk+lang8/pattern.json
    --regularization-weight 1.0
    

============================== English Explainable BART ==============================
explainable_bart/preprocess/eng/expect/valid/bin_temp
    --save-dir explainable_bart/exps/eng/temp
    --user-dir explainable_bart
    --task explainable_gec
    --arch egec_bart_large
    --restore-file ../../../resources/bart.large/model.pt
    --max-tokens 2048
    --optimizer adam
    --layernorm-embedding
    --weight-decay 0.01
    --share-all-embeddings
    --share-decoder-input-output-embed
    --update-freq 1
    --lr 3e-05
    --warmup-updates 100
    -s src
    -t tgt
    --dropout 0.3
    --lr-scheduler inverse_sqrt
    --clip-norm 0.1
    --criterion explainable_label_smoothed_cross_entropy
    --label-smoothing 0.1
    --explanation-weight 2.0
    --max-epoch 50
    --patience 10
    --adam-betas (0.9,0.999)
    --log-format tqdm
    --reset-lr-scheduler
    --reset-optimizer
    --reset-meters
    --reset-dataloader
    --fp16
    --skip-invalid-size-inputs-valid-test
    --find-unused-parameters
    --keep-last-epochs 1
    --eval-gec
    --eval-gec-min-update 100
    --eval-gec-metric errant_eng
    --eval-gec-output-prefix explainable_bart/exps/eng/temp/results/output
    --eval-gec-m2-filepath ../../../datasets/GEC/EGEC/expect/valid/expect_valid.errant
    --eval-gec-raw-filepath ../../../datasets/GEC/EGEC/expect/json/valid.json
    --eval-gec-exp-filepath explainable_bart/preprocess/eng/expect/valid/valid.bpe.exp
    --beam 5
    --bpe gpt2
    --gpt2-encoder-json explainable_bart/preprocess/eng/encoder.json
    --gpt2-vocab-bpe explainable_bart/preprocess/eng/vocab.bpe
    --remove-bpe
    --min-len 0
    --left-pad-source
    --explanation-format evidence-type
    --explanation-setting rationalization
    --explanation-before

============================== English Explainable BART Finetune ==============================
explainable_bart/preprocess/eng/expect_denoise/valid/bin
    --save-dir explainable_bart/exps/eng/expect_denoise/temp
    --user-dir explainable_bart
    --task explainable_gec
    --arch egec_bart_large
    --finetune-from-model explainable_bart/exps/eng/expect_denoise/expect_denoise-rationalization_evidence_type_after-enc_mlp-ew1.0_eps0/checkpoint_best_score.pt
    --max-tokens 2048
    --optimizer adam
    --layernorm-embedding
    --weight-decay 0.01
    --share-all-embeddings
    --share-decoder-input-output-embed
    --update-freq 1
    --lr 3e-05
    --warmup-updates 50
    -s src
    -t tgt
    --dropout 0.3
    --lr-scheduler inverse_sqrt
    --clip-norm 0.1
    --criterion explainable_label_smoothed_cross_entropy
    --label-smoothing 0.1
    --max-epoch 50
    --patience 10
    --adam-betas (0.9,0.999)
    --log-format tqdm
    --fp16
    --skip-invalid-size-inputs-valid-test
    --find-unused-parameters
    --keep-last-epochs 1
    --eval-gec
    --eval-gec-min-update 1
    --eval-gec-metric errant_eng
    --eval-gec-output-prefix explainable_bart/exps/eng/temp/results/output
    --eval-gec-m2-filepath ../../../datasets/GEC/EGEC/expect/denoise/valid/expect_valid_denoise.errant
    --eval-gec-raw-filepath ../../../datasets/GEC/EGEC/expect/denoise/valid/expect_valid_denoise.json
    --eval-gec-exp-filepath explainable_bart/preprocess/eng/expect_denoise/valid/valid.bpe.exp
    --beam 5
    --bpe gpt2
    --gpt2-encoder-json bart/preprocess/eng/encoder.json
    --gpt2-vocab-bpe bart/preprocess/eng/vocab.bpe
    --remove-bpe
    --min-len 0
    --left-pad-source
    --explanation-format evidence-type
    --explanation-setting rationalization
    --use-encoder-mlp
    --sequence-tagging
    --tagging-weight 2.0
    
"""
