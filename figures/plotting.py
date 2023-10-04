import matplotlib.pyplot as plt

from configs.args import TrainingArgs


def plot_first_batch(batch, args: TrainingArgs, collator_args):
    if args.train_type == "encoder":
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))
        prosody = (
            batch[0]
            .squeeze(1)
            .transpose(0, 1)
            .reshape(collator_args.enc_max_length, -1)
            .numpy()
            .T
        )
        phones = (
            batch[1].transpose(0, 1).reshape(collator_args.enc_max_length, -1).numpy().T
        )
        speaker = (
            batch[2].transpose(0, 1).reshape(collator_args.enc_max_length, -1).numpy().T
        )
        mask = (
            batch[3]
            .squeeze(1)
            .transpose(0, 1)
            .reshape(collator_args.enc_max_length, -1)
            .numpy()
            .T
        )
        axes[0].imshow(prosody, aspect="auto")
        axes[1].imshow(phones, aspect="auto")
        axes[2].imshow(speaker, aspect="auto")
        axes[3].imshow(mask, aspect="auto")
        axes[0].set_title("Prosody")
        axes[1].set_title("Phones")
        axes[2].set_title("Speaker")
        axes[3].set_title("Mask")
    elif args.train_type == "decoder":
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 10))
        prosody = (
            batch[0].transpose(0, 1).reshape(collator_args.dec_max_length, -1).numpy().T
        )
        phones = (
            batch[1].transpose(0, 1).reshape(collator_args.dec_max_length, -1).numpy().T
        )
        speaker = (
            batch[2].transpose(0, 1).reshape(collator_args.dec_max_length, -1).numpy().T
        )
        mel = (
            batch[3]
            .squeeze(1)
            .transpose(0, 1)
            .reshape(collator_args.dec_max_length, -1)
            .numpy()
            .T
        )
        mask = (
            batch[4]
            .squeeze(1)
            .transpose(0, 1)
            .reshape(collator_args.dec_max_length, -1)
            .numpy()
            .T
        )
        axes[0].imshow(prosody, aspect="auto")
        axes[1].imshow(phones, aspect="auto")
        axes[2].imshow(speaker, aspect="auto")
        axes[3].imshow(mel, aspect="auto")
        axes[4].imshow(mask, aspect="auto")
        axes[0].set_title("Prosody")
        axes[1].set_title("Phones")
        axes[2].set_title("Speaker")
        axes[3].set_title("Mel")
        axes[4].set_title("Mask")
    plt.tight_layout()
    return fig
