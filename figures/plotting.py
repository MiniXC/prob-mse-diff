import matplotlib.pyplot as plt

from configs.args import TrainingArgs


def plot_first_batch(batch, args: TrainingArgs, collator_args):
    if args.train_type == "encoder":
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        prosody = (
            batch[0].transpose(0, 1).reshape(collator_args.enc_max_length, -1).numpy().T
        )
        phones = (
            batch[1].transpose(0, 1).reshape(collator_args.enc_max_length, -1).numpy().T
        )
        speaker = (
            batch[2].transpose(0, 1).reshape(collator_args.enc_max_length, -1).numpy().T
        )
        axes[0].imshow(prosody, aspect="auto")
        axes[1].imshow(phones, aspect="auto")
        axes[2].imshow(speaker, aspect="auto")
        axes[0].set_title("Prosody")
        axes[1].set_title("Phones")
        axes[2].set_title("Speaker")
    elif args.train_type == "decoder":
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))
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
            batch[3].transpose(0, 1).reshape(collator_args.dec_max_length, -1).numpy().T
        )
        axes[0].imshow(prosody, aspect="auto")
        axes[1].imshow(phones, aspect="auto")
        axes[2].imshow(speaker, aspect="auto")
        axes[3].imshow(mel, aspect="auto")
    plt.tight_layout()
    return fig
