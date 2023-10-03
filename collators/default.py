import torch
from torch import nn
import numpy as np
from PIL import Image
from scipy.signal import cwt, ricker

from configs.args import EncoderCollatorArgs, DecoderCollatorArgs

WAVELET_WIDTHS = np.arange(1, 10, 1)


class EncoderCollator:
    def __init__(self, args: EncoderCollatorArgs):
        self.max_length = args.enc_max_length
        self.pack_factor = args.enc_pack_factor
        self.verbose = args.enc_verbose

    @staticmethod
    def compute_cwt(duration_array):
        cwt_matrix = cwt(duration_array, ricker, WAVELET_WIDTHS)
        return cwt_matrix.real

    @staticmethod
    def item_to_arrays(item):
        prosody = Image.open(item["prosody"])
        prosody = np.array(prosody)
        durs = prosody[-1]
        durs = (durs - durs.mean()) / durs.std()
        cwt_result = EncoderCollator.compute_cwt(durs)
        cwt_result = (
            (cwt_result - cwt_result.min())
            / (cwt_result.max() - cwt_result.min())
            * 255
        )
        cwt_result = cwt_result.astype(np.uint8)
        speaker = np.array(Image.open(item["speaker_utterance"]))
        prosody = np.concatenate((prosody, cwt_result, speaker), axis=0)
        # to float
        prosody = (prosody.astype(np.float32) / 255.0 - 0.5).T
        # speaker
        speaker = np.load(item["mean_speaker"])[np.newaxis, :]
        # repeat speaker to match prosody
        speaker = np.repeat(speaker, prosody.shape[0], axis=0)
        return prosody, np.load(item["phones"]), speaker

    def pack(self, prosody, phones, speaker):
        reverse = np.random.rand() > 0.5
        lengths = np.array([arr.shape[0] for arr in prosody])
        lengths_index = lengths.argsort()
        if reverse:
            lengths_index = lengths_index[::-1]
        prosody = [prosody[i] for i in lengths_index]
        phones = [phones[i] for i in lengths_index]
        speaker = [speaker[i] for i in lengths_index]
        packed_prosody = torch.zeros(
            len(prosody) // self.pack_factor, self.max_length, prosody[0].shape[1]
        )
        packed_phones = torch.zeros(
            len(phones) // self.pack_factor, self.max_length
        ).long()
        packed_speaker = torch.zeros(
            len(speaker) // self.pack_factor, self.max_length, speaker[0].shape[1]
        )
        num_cut = 0
        num_pad = 0
        for i in range(len(prosody) // self.pack_factor):
            pack_arrs_prosody = [prosody[i + j] for j in range(self.pack_factor // 2)]
            pack_arrs_prosody += [
                prosody[-(i + j + 1)] for j in range(self.pack_factor // 2)
            ]
            pack_arrs_phones = [phones[i + j] for j in range(self.pack_factor // 2)]
            pack_arrs_phones += [
                phones[-(i + j + 1)] for j in range(self.pack_factor // 2)
            ]
            pack_arrs_speaker = [speaker[i + j] for j in range(self.pack_factor // 2)]
            pack_arrs_speaker += [
                speaker[-(i + j + 1)] for j in range(self.pack_factor // 2)
            ]
            current_len = 0
            for j, arr in enumerate(pack_arrs_prosody):
                if current_len + arr.shape[0] > self.max_length:
                    cut_last_by = current_len + arr.shape[0] - self.max_length
                    arr = arr[:-cut_last_by]
                    arr_phones = pack_arrs_phones[j][:-cut_last_by]
                    arr_speaker = pack_arrs_speaker[j][:-cut_last_by]
                    packed_prosody[i, current_len:, :] = torch.from_numpy(arr)
                    packed_phones[i, current_len:] = torch.from_numpy(
                        arr_phones.astype(np.int32)
                    )
                    packed_speaker[i, current_len:, :] = torch.from_numpy(arr_speaker)
                    num_cut += cut_last_by
                    current_len = self.max_length
                    break
                else:
                    packed_prosody[
                        i, current_len : (current_len + arr.shape[0]), :
                    ] = torch.from_numpy(arr)
                    packed_phones[
                        i, current_len : (current_len + arr.shape[0])
                    ] = torch.from_numpy(pack_arrs_phones[j].astype(np.int32))
                    packed_speaker[
                        i, current_len : (current_len + arr.shape[0]), :
                    ] = torch.from_numpy(pack_arrs_speaker[j])
                    current_len += arr.shape[0]
            num_pad += self.max_length - current_len
        pct_cut = num_cut / (self.max_length * len(prosody) // self.pack_factor)
        pct_pad = num_pad / (self.max_length * len(prosody) // self.pack_factor)
        if self.verbose:
            print(f"Cut {pct_cut:.2%} of the data")
            print(f"Pad {pct_pad:.2%} of the data")
        return packed_prosody, packed_phones, packed_speaker

    def __call__(self, batch):
        items = [EncoderCollator.item_to_arrays(item) for item in batch]
        prosody, phones, speaker = zip(*items)
        pack_sequence = np.random.rand() > 0.05
        if pack_sequence:
            packed_prosody, packed_phones, packed_speaker = self.pack(
                prosody, phones, speaker
            )
        else:
            # use the first half of the batch and pad to self.max_length
            packed_prosody = torch.zeros(
                len(prosody) // self.pack_factor, self.max_length, prosody[0].shape[1]
            )
            packed_phones = torch.zeros(
                len(phones) // self.pack_factor, self.max_length
            ).long()
            packed_speaker = torch.zeros(
                len(speaker) // self.pack_factor, self.max_length, speaker[0].shape[1]
            )
            for i in range(len(prosody) // self.pack_factor):
                first_arr_prosody = prosody[i]
                first_arr_phones = phones[i]
                first_arr_speaker = speaker[i]
                first_len = first_arr_prosody.shape[0]
                if first_len > self.max_length:
                    first_arr_prosody = first_arr_prosody[: self.max_length]
                    first_arr_phones = first_arr_phones[: self.max_length]
                    first_arr_speaker = first_arr_speaker[: self.max_length]
                    packed_prosody[i, :, :] = torch.from_numpy(first_arr_prosody)
                    packed_phones[i, :] = torch.from_numpy(
                        first_arr_phones.astype(np.int32)
                    )
                    packed_speaker[i, :, :] = torch.from_numpy(first_arr_speaker)
                else:
                    packed_prosody[i, :first_len, :] = torch.from_numpy(
                        first_arr_prosody
                    )
                    packed_phones[i, :first_len] = torch.from_numpy(
                        first_arr_phones.astype(np.int32)
                    )
                    packed_speaker[i, :first_len, :] = torch.from_numpy(
                        first_arr_speaker
                    )
        packed_prosody = packed_prosody.unsqueeze(1)
        return packed_prosody, packed_phones, packed_speaker


class DecoderCollator:
    def __init__(self, args: DecoderCollatorArgs):
        self.max_length = args.dec_max_length
        self.pack_factor = args.dec_pack_factor
        self.verbose = args.dec_verbose

    @staticmethod
    def item_to_arrays(item):
        prosody, phones, speaker = EncoderCollator.item_to_arrays(item)
        duration = prosody[:, 30]
        # denormalize
        duration = np.round((duration + 0.5) * 50, 0).astype(np.int32)
        mel = np.array(Image.open(item["mel"])).T
        if mel.shape[0] > duration.sum():
            duration[-1] += mel.shape[0] - duration.sum()
        # repeat prosody to match mel using duration
        prosody = np.repeat(prosody, duration, axis=0)
        # repeat speaker to match mel using duration
        speaker = np.repeat(speaker, duration, axis=0)
        # repeat phones to match mel using duration
        phones = np.repeat(phones, duration, axis=0)
        return prosody, phones, speaker, mel

    def pack(self, prosody, phones, speaker, mel):
        reverse = np.random.rand() > 0.5
        lengths = np.array([arr.shape[0] for arr in prosody])
        lengths_index = lengths.argsort()
        if reverse:
            lengths_index = lengths_index[::-1]
        prosody = [prosody[i] for i in lengths_index]
        phones = [phones[i] for i in lengths_index]
        speaker = [speaker[i] for i in lengths_index]
        mel = [mel[i] for i in lengths_index]
        packed_prosody = torch.zeros(
            len(prosody) // self.pack_factor, self.max_length, prosody[0].shape[1]
        )
        packed_phones = torch.zeros(
            len(phones) // self.pack_factor, self.max_length
        ).long()
        packed_speaker = torch.zeros(
            len(speaker) // self.pack_factor, self.max_length, speaker[0].shape[1]
        )
        packed_mel = torch.zeros(
            len(mel) // self.pack_factor, self.max_length, mel[0].shape[1]
        )
        num_cut = 0
        num_pad = 0
        for i in range(len(prosody) // self.pack_factor):
            pack_arrs_prosody = [prosody[i + j] for j in range(self.pack_factor // 2)]
            pack_arrs_prosody += [
                prosody[-(i + j + 1)] for j in range(self.pack_factor // 2)
            ]
            pack_arrs_phones = [phones[i + j] for j in range(self.pack_factor // 2)]
            pack_arrs_phones += [
                phones[-(i + j + 1)] for j in range(self.pack_factor // 2)
            ]
            pack_arrs_speaker = [speaker[i + j] for j in range(self.pack_factor // 2)]
            pack_arrs_speaker += [
                speaker[-(i + j + 1)] for j in range(self.pack_factor // 2)
            ]
            pack_arrs_mel = [mel[i + j] for j in range(self.pack_factor // 2)]
            pack_arrs_mel += [mel[-(i + j + 1)] for j in range(self.pack_factor // 2)]
            current_len = 0
            for j, arr in enumerate(pack_arrs_prosody):
                if current_len + arr.shape[0] > self.max_length:
                    cut_last_by = current_len + arr.shape[0] - self.max_length
                    arr = arr[:-cut_last_by]
                    arr_phones = pack_arrs_phones[j][:-cut_last_by]
                    arr_speaker = pack_arrs_speaker[j][:-cut_last_by]
                    arr_mel = pack_arrs_mel[j][:-cut_last_by]
                    packed_prosody[i, current_len:, :] = torch.from_numpy(arr)
                    packed_phones[i, current_len:] = torch.from_numpy(
                        arr_phones.astype(np.int32)
                    )
                    packed_speaker[i, current_len:, :] = torch.from_numpy(arr_speaker)
                    packed_mel[i, current_len:, :] = torch.from_numpy(arr_mel)
                    num_cut += cut_last_by
                    current_len = self.max_length
                    break
                else:
                    packed_prosody[
                        i, current_len : (current_len + arr.shape[0]), :
                    ] = torch.from_numpy(arr)
                    packed_phones[
                        i, current_len : (current_len + arr.shape[0])
                    ] = torch.from_numpy(pack_arrs_phones[j].astype(np.int32))
                    packed_speaker[
                        i, current_len : (current_len + arr.shape[0]), :
                    ] = torch.from_numpy(pack_arrs_speaker[j])
                    packed_mel[
                        i, current_len : (current_len + arr.shape[0]), :
                    ] = torch.from_numpy(pack_arrs_mel[j])
                    current_len += arr.shape[0]
            num_pad += self.max_length - current_len
        pct_cut = num_cut / (self.max_length * len(prosody) // self.pack_factor)
        pct_pad = num_pad / (self.max_length * len(prosody) // self.pack_factor)
        if self.verbose:
            print(f"Cut {pct_cut:.2%} of the data")
            print(f"Pad {pct_pad:.2%} of the data")
        return packed_prosody, packed_phones, packed_speaker, packed_mel

    def __call__(self, batch):
        items = [DecoderCollator.item_to_arrays(item) for item in batch]
        prosody, phones, speaker, mel = zip(*items)
        pack_sequence = np.random.rand() > 0.05
        if pack_sequence:
            packed_prosody, packed_phones, packed_speaker, packed_mel = self.pack(
                prosody, phones, speaker, mel
            )
        else:
            # use the first half of the batch and pad to self.max_length
            packed_prosody = torch.zeros(
                len(prosody) // self.pack_factor, self.max_length, prosody[0].shape[1]
            )
            packed_phones = torch.zeros(
                len(phones) // self.pack_factor, self.max_length
            ).long()
            packed_speaker = torch.zeros(
                len(speaker) // self.pack_factor, self.max_length, speaker[0].shape[1]
            )
            packed_mel = torch.zeros(
                len(mel) // self.pack_factor, self.max_length, mel[0].shape[1]
            )
            for i in range(len(prosody) // self.pack_factor):
                first_arr_prosody = prosody[i]
                first_arr_phones = phones[i]
                first_arr_speaker = speaker[i]
                first_arr_mel = mel[i]
                first_len = first_arr_prosody.shape[0]
                if first_len > self.max_length:
                    first_arr_prosody = first_arr_prosody[: self.max_length]
                    first_arr_phones = first_arr_phones[: self.max_length]
                    first_arr_speaker = first_arr_speaker[: self.max_length]
                    first_arr_mel = first_arr_mel[: self.max_length]
                    packed_prosody[i, :, :] = torch.from_numpy(first_arr_prosody)
                    packed_phones[i, :] = torch.from_numpy(
                        first_arr_phones.astype(np.int32)
                    )
                    packed_speaker[i, :, :] = torch.from_numpy(first_arr_speaker)
                    packed_mel[i, :, :] = torch.from_numpy(first_arr_mel)
                else:
                    packed_prosody[i, :first_len, :] = torch.from_numpy(
                        first_arr_prosody
                    )
                    packed_phones[i, :first_len] = torch.from_numpy(
                        first_arr_phones.astype(np.int32)
                    )
                    packed_speaker[i, :first_len, :] = torch.from_numpy(
                        first_arr_speaker
                    )
                    packed_mel[i, :first_len, :] = torch.from_numpy(first_arr_mel)
        packed_mel = packed_mel.unsqueeze(1)
        return packed_prosody, packed_phones, packed_speaker, packed_mel
