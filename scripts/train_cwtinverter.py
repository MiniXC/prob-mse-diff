import os
import sys
from collections import deque
from pathlib import Path

sys.path.append(".")  # add root of project to path

# torch & hf
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    get_linear_schedule_with_warmup,
    HfArgumentParser,
)
from diffusers.optimization import get_scheduler
from datasets import load_dataset
import torch.nn.functional as F

# logging & etc
from torchinfo import summary
from torchview import draw_graph
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import yaml
from rich.console import Console
from PIL import Image
import torchaudio

# plotting
import matplotlib.pyplot as plt
from figures.plotting import plot_first_batch

console = Console()

# local imports
from configs.args import (
    TrainingArgs,
    EncoderCollatorArgs,
)
from configs.cwtargs import ModelArgs
from configs.validation import validate_args
from util.remote import wandb_update_config, wandb_init, push_to_hub
from model.cwt_inverter import CWTInverterModel
from collators import get_collator

def print_and_draw_model(pack_factor):
    bsz = training_args.batch_size // pack_factor
    dummy_input = model.dummy_input
    # repeat dummy input to match batch size (regardless of how many dimensions)
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.repeat((bsz,) + (1,) * (len(dummy_input.shape) - 1))
        console_print(f"[green]input shape[/green]: {dummy_input.shape}")
    elif isinstance(dummy_input, list):
        dummy_input = [
            x.repeat((bsz,) + (1,) * (len(x.shape) - 1)) for x in dummy_input
        ]
        console_print(f"[green]input shapes[/green]: {[x.shape for x in dummy_input]}")
    model_summary = summary(
        model,
        input_data=dummy_input,
        verbose=0,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
        ],
    )
    console_print(model_summary)
    if accelerator.is_main_process:
        model_graph = draw_graph(
            model,
            input_data=dummy_input,
            save_graph=True,
            directory="figures/",
            filename="model",
            expand_nested=True,
        )


def console_print(*args, **kwargs):
    if accelerator.is_main_process:
        console.print(*args, **kwargs)


def console_rule(*args, **kwargs):
    if accelerator.is_main_process:
        console.rule(*args, **kwargs)


def wandb_log(prefix, log_dict, round_n=3, print_log=True):
    if accelerator.is_main_process:
        log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
        wandb.log(log_dict, step=global_step)
        if print_log:
            log_dict = {
                k: round(v, round_n) for k, v in log_dict.items() if "image" not in k
            }
            console.log(log_dict)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(name_override=None):
    accelerator.wait_for_everyone()
    checkpoint_name = training_args.run_name
    if name_override is not None:
        name = name_override
    else:
        name = f"step_{global_step}"
    checkpoint_path = Path(training_args.checkpoint_path) / checkpoint_name / name
    if name_override is None:
        # remove old checkpoints
        if checkpoint_path.exists():
            for f in checkpoint_path.iterdir():
                os.remove(f)
    # model
    model.save_model(checkpoint_path, accelerator)
    if accelerator.is_main_process:
        # training args
        with open(checkpoint_path / "training_args.yml", "w") as f:
            f.write(yaml.dump(training_args.__dict__, Dumper=yaml.Dumper))
        # collator args
        with open(checkpoint_path / "collator_args.yml", "w") as f:
            f.write(yaml.dump(collator_args.__dict__, Dumper=yaml.Dumper))
        if training_args.push_to_hub:
            push_to_hub(
                training_args.hub_repo,
                checkpoint_path,
                commit_message=f"step {global_step}",
            )
    accelerator.wait_for_everyone()
    return checkpoint_path


def train_epoch(epoch):
    global global_step
    model.train()
    losses = deque(maxlen=training_args.log_every_n_steps)
    step = 0
    console_rule(f"Epoch {epoch}")
    last_loss = None
    for batch in train_dl:
        with accelerator.accumulate(model):
            (
                packed_prosody,
                packed_phones,
                packed_speaker,
                packed_mask,
                lm_inputs,
            ) = batch
            bsz = packed_prosody.shape[0]
            packed_prosody = packed_prosody.squeeze(1)
            tgt_durations = packed_prosody[:, :, 30]
            packed_mask = packed_mask.squeeze(1)
            # tgt_durations_mean = (tgt_durations * packed_mask) / packed_mask.sum()
            # tgt_durations_std = torch.sqrt(
            #     ((tgt_durations - tgt_durations_mean) ** 2 * packed_mask)
            #     / packed_mask.sum()
            # ) + 1e-8
            # tgt_durations = (tgt_durations - tgt_durations_mean) / tgt_durations_std
            # the same but batched
            tgt_durations_mean = (tgt_durations * packed_mask).sum(1) / packed_mask.sum(1)
            tgt_durations_mean = tgt_durations_mean.unsqueeze(1)
            tgt_durations_std = torch.sqrt(
                ((tgt_durations - tgt_durations_mean) ** 2 * packed_mask).sum(1)
                / packed_mask.sum(1)
            ) + 1e-8
            tgt_durations_std = tgt_durations_std.unsqueeze(1)
            tgt_durations = (tgt_durations - tgt_durations_mean) / tgt_durations_std
            input_cwt = packed_prosody[:, :, 31:31+model_args.n_widths]
            output = model(
                input_cwt,
            )
            loss = F.mse_loss(output.transpose(1, 2).squeeze(1), tgt_durations, reduction="none")
            loss = loss * packed_mask
            loss = loss.sum() / packed_mask.sum()
            losses.append(loss.detach())
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                model.parameters(), training_args.gradient_clip_val
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if (
            step > 0
            and step % training_args.log_every_n_steps == 0
            and accelerator.is_main_process
        ):
            last_loss = torch.mean(torch.tensor(losses)).item()
            wandb_log("train", {"loss": last_loss}, print_log=False)
        if (
            training_args.do_save
            and global_step > 0
            and global_step % training_args.save_every_n_steps == 0
        ):
            save_checkpoint()
        if training_args.n_steps is not None and global_step >= training_args.n_steps:
            return
        if (
            training_args.eval_every_n_steps is not None
            and global_step > 0
            and global_step % training_args.eval_every_n_steps == 0
        ):
            if training_args.do_full_eval:
                evaluate()
            else:
                evaluate_loss_only()
            console_rule(f"Epoch {epoch}")
        step += 1
        global_step += 1
        if accelerator.is_main_process:
            pbar.update(1)
            if last_loss is not None:
                pbar.set_postfix({"loss": f"{last_loss:.3f}"})


def evaluate():
    # pass the first batch through the pipeline
    checkpoint_path = save_checkpoint("temp")
    if accelerator.is_main_process:
        eval_model = CWTInverterModel.from_pretrained(checkpoint_path)
        eval_model = eval_model.eval()
        eval_model = eval_model.to("cpu")
        mses = []
        losses = []
        i = 0
        for batch in tqdm(val_dl, desc="eval"):
            (
                packed_prosody,
                packed_phones,
                packed_speaker,
                packed_mask,
                lm_inputs,
            ) = batch
            bsz = packed_prosody.shape[0]
            packed_prosody = packed_prosody.squeeze(1).to("cpu")
            tgt_durations = packed_prosody[:, :, 30]
            packed_mask = packed_mask.squeeze(1).to("cpu")
            # tgt_durations_mean = (tgt_durations * packed_mask) / packed_mask.sum()
            # tgt_durations_std = torch.sqrt(
            #     ((tgt_durations - tgt_durations_mean) ** 2 * packed_mask)
            #     / packed_mask.sum()
            # ) + 1e-8
            # tgt_durations = (tgt_durations - tgt_durations_mean) / tgt_durations_std
            # the same but batched
            tgt_durations_mean = (tgt_durations * packed_mask).sum(1) / packed_mask.sum(1)
            tgt_durations_mean = tgt_durations_mean.unsqueeze(1)
            tgt_durations_std = torch.sqrt(
                ((tgt_durations - tgt_durations_mean) ** 2 * packed_mask).sum(1)
                / packed_mask.sum(1)
            ) + 1e-8
            tgt_durations_std = tgt_durations_std.unsqueeze(1)
            tgt_durations = (tgt_durations - tgt_durations_mean) / tgt_durations_std
            input_cwt = packed_prosody[:, :, 31:31+model_args.n_widths]
            output = eval_model(
                input_cwt,
            )
            loss = F.mse_loss(output.transpose(1, 2).squeeze(1), tgt_durations, reduction="none")
            loss = loss * packed_mask
            loss = loss.sum() / packed_mask.sum()
            losses.append(loss.detach())
            tgt_durations = tgt_durations.detach().cpu()
            packed_mask = packed_mask.detach().cpu()
            output = output.squeeze(-1)
            # mask both
            output = output
            tgt_durations = tgt_durations
            # denormalize
            output = output * tgt_durations_std + tgt_durations_mean
            tgt_durations = tgt_durations * tgt_durations_std + tgt_durations_mean
            output = torch.concatenate(
                [o[packed_mask[i].bool()] for i, o in enumerate(output)], dim=0
            )
            tgt_durations = torch.concatenate(
                [o[packed_mask[i].bool()] for i, o in enumerate(tgt_durations)], dim=0
            )
            output_c, tgt_c = torch.round(2**(output*11)), torch.round(2**(tgt_durations*11))
            if i == 0:
                print(output_c, tgt_c)
            # compute mse
            mse = ((output_c - tgt_c)**2).mean().detach().numpy()
            mses.append(mse)
            i += 1
        # log
        wandb_log(
            "val",
            {
                "loss": torch.mean(torch.tensor(losses)).item(),
                "mse": np.mean(mses),
            },
        )


def main():
    global accelerator, training_args, model_args, collator_args, train_dl, val_dl, optimizer, scheduler, model, global_step, pbar

    global_step = 0

    parser = HfArgumentParser(
        [TrainingArgs, ModelArgs, EncoderCollatorArgs]
    )

    # parse args
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yml"):
        with open(sys.argv[1], "r") as f:
            args_dict = yaml.load(f, Loader=yaml.Loader)
        # additonally parse args from command line
        (
            training_args,
            model_args,
            enc_collator_args,
        ) = parser.parse_args_into_dataclasses(sys.argv[2:])
        # update args from yaml
        for k, v in args_dict.items():
            if hasattr(training_args, k):
                setattr(training_args, k, v)
            if hasattr(model_args, k):
                setattr(model_args, k, v)
            if hasattr(enc_collator_args, k):
                setattr(enc_collator_args, k, v)
        if len(sys.argv) > 2:
            console_print(
                f"[yellow]WARNING[/yellow]: yaml args will be override command line args"
            )
    else:
        (
            training_args,
            model_args,
            enc_collator_args,
        ) = parser.parse_args_into_dataclasses()

    if not training_args.bf16:
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    if training_args.train_type == "encoder":
        collator_args = enc_collator_args
        max_length = collator_args.enc_max_length
        model_args.max_length = max_length
        collator_args.lm_condition = None
    else:
        raise ValueError(f"train_type {training_args.train_type} not supported")

    if training_args.train_type == "encoder":
        pack_factor = collator_args.enc_pack_factor

    # check if run name is specified
    if training_args.run_name is None:
        raise ValueError("run_name must be specified")
    if (
        training_args.do_save
        and (Path(training_args.checkpoint_path) / training_args.run_name).exists()
    ):
        raise ValueError(f"run_name {training_args.run_name} already exists")

    # wandb
    if accelerator.is_main_process:
        wandb_name, wandb_project, wandb_dir, wandb_mode = (
            training_args.run_name,
            training_args.wandb_project,
            training_args.wandb_dir,
            training_args.wandb_mode,
        )
        wandb_init(wandb_name, wandb_project, wandb_dir, wandb_mode)
        wandb.run.log_code()

    # log args
    console_rule("Arguments")
    console_print(training_args)
    console_print(model_args)
    console_print(collator_args)
    if accelerator.is_main_process:
        wandb_update_config(
            {
                "training": training_args,
                "model": model_args,
                "collator": collator_args,
            }
        )
    validate_args(training_args, model_args, collator_args)

    # Distribution Information
    console_rule("Distribution Information")
    console_print(f"[green]accelerator[/green]: {accelerator}")
    console_print(f"[green]n_procs[/green]: {accelerator.num_processes}")
    console_print(f"[green]process_index[/green]: {accelerator.process_index}")

    # model
    model = CWTInverterModel(model_args)
    if training_args.load_from_checkpoint is not None:
        console_print(
            f"[green]load_from_checkpoint[/green]: {training_args.load_from_checkpoint}"
        )
        model = CWTInverterModel.from_pretrained(training_args.load_from_checkpoint)
    console_rule("Model")
    print_and_draw_model(pack_factor)

    # dataset
    console_rule("Dataset")

    console_print(f"[green]dataset[/green]: {training_args.dataset}")
    console_print(f"[green]train_split[/green]: {training_args.train_split}")
    console_print(f"[green]val_split[/green] will be generated from train_split")

    train_ds = load_dataset(training_args.dataset, split=training_args.train_split)
    np.random.seed(training_args.seed)
    speakers = np.random.choice(
        train_ds.unique("speaker_id"),
        size=training_args.speakers_in_validation,
        replace=False,
    )
    # get training_args.samples_per_speaker_in_validation samples per speaker
    val_ds = train_ds.filter(
        lambda x: x["speaker_id"] in speakers and np.random.rand() < 0.5,
        keep_in_memory=True,
    )
    val_ids = val_ds.unique("id")
    # remove val_ds from train_ds
    train_ds = train_ds.filter(lambda x: x["id"] not in val_ids, keep_in_memory=True)

    val_ds_unseen = load_dataset(
        training_args.dataset,
        split=training_args.unseen_validation_split,
        keep_in_memory=True,
    )

    print(f"train_ds: {len(train_ds)}")
    print(f"val_ds: {len(val_ds)}")
    print(f"val_ds_unseen: {len(val_ds_unseen)}")

    console_print(f"[green]train[/green]: {len(train_ds)}")
    console_print(f"[green]val[/green]: {len(val_ds)}")

    # collator
    collator = get_collator(training_args, collator_args)

    # plot first batch
    if accelerator.is_main_process:
        first_batch = collator([train_ds[i] for i in range(training_args.batch_size)])
        plot_first_batch(first_batch, training_args, collator_args)
        plt.savefig("figures/first_batch.png")
        plt.close()

    # dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=training_args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=training_args.valid_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )

    # scheduler
    if training_args.lr_schedule == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.n_steps,
        )
    elif training_args.lr_schedule == "cosine":
        scheduler = get_scheduler(
            training_args.lr_schedule,
            optimizer=optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.n_steps,
        )


    # accelerator
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )

    # evaluation
    if training_args.eval_only:
        console_rule("Evaluation")
        seed_everything(training_args.seed)
        evaluate()
        return

    # training
    console_rule("Training")
    seed_everything(training_args.seed)
    pbar_total = training_args.n_steps
    training_args.n_epochs = training_args.n_steps // len(train_dl) + 1
    console_print(f"[green]n_epochs[/green]: {training_args.n_epochs}")
    console_print(
        f"[green]effective_batch_size[/green]: {training_args.batch_size*accelerator.num_processes//pack_factor}"
    )
    pbar = tqdm(total=pbar_total, desc="step")
    for i in range(training_args.n_epochs):
        train_epoch(i)
    console_rule("Evaluation")
    seed_everything(training_args.seed)

    evaluate()

    # save final model
    console_rule("Saving")
    if training_args.do_save:
        save_checkpoint()

    # wandb sync reminder
    if accelerator.is_main_process and training_args.wandb_mode == "offline":
        console_rule("Weights & Biases")
        console_print(
            f"use \n[magenta]wandb sync {Path(wandb.run.dir).parent}[/magenta]\nto sync offline run"
        )


if __name__ == "__main__":
    main()
