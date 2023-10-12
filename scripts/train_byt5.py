import os
import sys
from collections import deque
from pathlib import Path

sys.path.append(".")  # add root of project to path

# torch & hf
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, HfArgumentParser
from datasets import load_dataset
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer, Adafactor
from torchmetrics.text import CharErrorRate

# logging & etc
from torchinfo import summary
from torchview import draw_graph
import wandb
import numpy as np
from tqdm.auto import tqdm
import yaml
from rich.console import Console

# plotting
import matplotlib.pyplot as plt
from figures.plotting import plot_first_batch_byt5 as plot_first_batch

console = Console()

# local imports
from configs.byt5args import (
    TrainingArgs,
    CollatorArgs,
    ModelArgs,
)
from configs.validation import validate_args
from util.remote import wandb_update_config, wandb_init, push_to_hub
from collators import get_collator
from model.byt5_wrapper import ByT5Wrapper


def print_and_draw_model():
    bsz = training_args.batch_size
    dummy_input = "Hello, my dog is cute"
    dummy_input = AutoTokenizer.from_pretrained("google/byt5-small")(
        dummy_input, return_tensors="pt"
    )["input_ids"]
    dummy_decoder_input = "My dog is cute"
    dummy_decoder_input = AutoTokenizer.from_pretrained("google/byt5-small")(
        dummy_decoder_input, return_tensors="pt"
    )["input_ids"]
    dummy_input = [
        dummy_input,
        dummy_decoder_input,
    ]
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
        input_data={
            "input_ids": dummy_input[0],
            "labels": dummy_input[1],
            "speaker": torch.randn(bsz, 256),
        },
        verbose=0,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
        ],
    )
    console_print(model_summary)
    if accelerator.is_main_process:
        _ = draw_graph(
            model,
            input_data={
                "input_ids": dummy_input[0],
                "labels": dummy_input[1],
                "speaker": torch.randn(bsz, 256),
            },
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
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            speaker = batch["speaker"]
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch["labels"],
                speaker=speaker,
            ).loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                model.parameters(), training_args.gradient_clip_val
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        losses.append(loss.detach())
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
        print(checkpoint_path)
        model = ByT5Wrapper.from_pretrained(checkpoint_path)
        device = "cpu"
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        with torch.no_grad():
            batch = next(iter(val_dl))
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            speaker = batch["speaker"].to(device)
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, speaker=speaker
            )
            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gt_outputs = tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
            cer = CharErrorRate()
            cer_val = cer(outputs, gt_outputs)
            wandb_log(
                "val",
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "gt_outputs": gt_outputs,
                    "cer": cer_val,
                },
                print_log=False,
            )
            console_print(f"[green]CER[/green]: {cer_val}")
            # print first 5
            for i in range(5):
                console_print(f"[green]input[/green]: {inputs[i]}")
                console_print(f"[green]output[/green]: {outputs[i]}")
                console_print(f"[green]gt_output[/green]: {gt_outputs[i]}")
        evaluate_loss_only()


def evaluate_loss_only():
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dl), desc="val"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            speaker = batch["speaker"]
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speaker=speaker,
                labels=batch["labels"],
            ).loss
            losses.append(loss.detach())
    loss = torch.mean(torch.tensor(losses)).item()
    wandb_log("val", {"loss": loss}, print_log=False)


def main():
    global accelerator, training_args, model_args, collator_args, train_dl, val_dl, optimizer, scheduler, model, global_step, pbar

    global_step = 0

    parser = HfArgumentParser(
        [
            TrainingArgs,
            CollatorArgs,
            ModelArgs,
        ]
    )

    accelerator = Accelerator()

    # parse args
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yml"):
        with open(sys.argv[1], "r") as f:
            args_dict = yaml.load(f, Loader=yaml.Loader)
        # additonally parse args from command line
        (
            training_args,
            collator_args,
            model_args,
        ) = parser.parse_args_into_dataclasses(sys.argv[2:])
        # update args from yaml
        for k, v in args_dict.items():
            if hasattr(training_args, k):
                setattr(training_args, k, v)
            if hasattr(collator_args, k):
                setattr(collator_args, k, v)
            if hasattr(model_args, k):
                setattr(model_args, k, v)
        if len(sys.argv) > 2:
            console_print(
                f"[yellow]WARNING[/yellow]: yaml args will override command line args"
            )
    else:
        (
            training_args,
            collator_args,
            model_args,
        ) = parser.parse_args_into_dataclasses()

    if training_args.train_type != "byt5":
        raise ValueError(f"train_type {training_args.train_type} not supported")

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
    console_print(collator_args)
    if accelerator.is_main_process:
        wandb_update_config(
            {
                "training": training_args,
                "collator": collator_args,
            }
        )
    validate_args(training_args, collator_args)

    # Distribution Information
    console_rule("Distribution Information")
    console_print(f"[green]accelerator[/green]: {accelerator}")
    console_print(f"[green]n_procs[/green]: {accelerator.num_processes}")
    console_print(f"[green]process_index[/green]: {accelerator.process_index}")

    # model
    if training_args.load_from_checkpoint is None:
        byt5_model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
        model = ByT5Wrapper(model_args, byt5_model)
        console_print(
            f"[green]load_from_checkpoint[/green]: {training_args.load_from_checkpoint}"
        )
    else:
        model = ByT5Wrapper.from_pretrained(training_args.load_from_checkpoint)
        console_print(
            f"[green]load_from_checkpoint[/green]: {training_args.load_from_checkpoint}"
        )
    console_rule("Model")
    print_and_draw_model()

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
    collator_val = get_collator(training_args, collator_args)
    # collator_val.inference = True

    # plot first batch
    if accelerator.is_main_process:
        first_batch = collator([train_ds[i] for i in range(training_args.batch_size)])
        plot_first_batch(first_batch)
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
        collate_fn=collator_val,
    )

    # optimizer
    if not training_args.use_adafactor:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.lr,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        )
    else:
        optimizer = Adafactor(
            model.parameters(),
            lr=training_args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=1e-6,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    # scheduler
    if training_args.lr_schedule == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.n_steps,
        )
    else:
        raise ValueError(f"lr_schedule {training_args.lr_schedule} not supported")

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
        f"[green]effective_batch_size[/green]: {training_args.batch_size*accelerator.num_processes}"
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
