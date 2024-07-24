from models import DurationNet, SynthesizerTrn
import soundfile as sf  # To save the audio file
import regex
import numpy as np
from types import SimpleNamespace
import unicodedata
import re
import json
import os
import argparse
import torch  # isort:skip

from vinorm import TTSnorm
torch.manual_seed(42)


def text_to_phone_idx(text):
    # unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # normalize text
    text = TTSnorm(text, punc=False, unknown=True, lower=True, rule=False)

    text = num_re.sub(r" \1 ", text)
    words = text.split()
    # words = [read_number(w) if num_re.fullmatch(w) else w for w in words]
    text = " ".join(words)

    # remove redundant spaces
    text = re.sub(r"\s+", " ", text)
    # remove leading and trailing spaces
    text = text.strip()
    # convert words to phone indices
    tokens = []
    for c in text:
        # if c is "," or ".", add <sil> phone
        if c in ":,.!?;(":
            tokens.append(sil_idx)
        elif c in phone_set:
            tokens.append(phone_set.index(c))
        elif c == " ":
            # add <sep> phone
            tokens.append(0)
    if tokens[0] != sil_idx:
        # insert <sil> phone at the beginning
        tokens = [sil_idx, 0] + tokens
    if tokens[-1] != sil_idx:
        tokens = tokens + [0, sil_idx]
    return tokens


def text_to_speech(duration_net, generator, text):
    # prevent too long text
    if len(text) > 500:
        text = text[:500]

    phone_idx = text_to_phone_idx(text)
    batch = {
        "phone_idx": np.array([phone_idx]),
        "phone_length": np.array([len(phone_idx)]),
    }

    # predict phoneme duration
    phone_length = torch.from_numpy(
        batch["phone_length"].copy()).long().to(device)
    phone_idx = torch.from_numpy(batch["phone_idx"].copy()).long().to(device)
    with torch.inference_mode():
        phone_duration = duration_net(phone_idx, phone_length)[:, :, 0] * 1000
    phone_duration = torch.where(
        phone_idx == sil_idx, torch.clamp_min(
            phone_duration, 200), phone_duration
    )
    phone_duration = torch.where(phone_idx == 0, 0, phone_duration)

    # generate waveform
    end_time = torch.cumsum(phone_duration, dim=-1)
    start_time = end_time - phone_duration
    start_frame = start_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    end_frame = end_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    spec_length = end_frame.max(dim=-1).values
    pos = torch.arange(0, spec_length.item(), device=device)
    attn = torch.logical_and(
        pos[None, :, None] >= start_frame[:, None, :],
        pos[None, :, None] < end_frame[:, None, :],
    ).float()
    with torch.inference_mode():
        y_hat = generator.infer(
            phone_idx, phone_length, spec_length, attn, max_len=None, noise_scale=0.667
        )[0]
    wave = y_hat[0, 0].data.cpu().numpy()
    return (wave * (2**15)).astype(np.int16)


def load_models():
    duration_net = DurationNet(hps.data.vocab_size, 64, 4).to(device)
    duration_net.load_state_dict(torch.load(
        duration_model_path, map_location=device))
    duration_net = duration_net.eval()
    generator = SynthesizerTrn(
        hps.data.vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
    ).to(device)
    del generator.enc_q
    ckpt = torch.load(lightspeed_model_path, map_location=device)
    params = {}
    for k, v in ckpt["net_g"].items():
        k = k[7:] if k.startswith("module.") else k
        params[k] = v
    generator.load_state_dict(params, strict=False)
    del ckpt, params
    generator = generator.eval()
    return duration_net, generator


def speak(text):
    duration_net, generator = load_models()
    paragraphs = text.split("\n")
    clips = []  # list of audio clips
    # silence = np.zeros(hps.data.sampling_rate // 4)
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph == "":
            continue
        clips.append(text_to_speech(duration_net, generator, paragraph))
        # clips.append(silence)
    y = np.concatenate(clips)
    return hps.data.sampling_rate, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input string and voice")
    parser.add_argument(
        '--gender', choices=['male', 'female'], help='Specify gender: male or female')
    parser.add_argument('--text', type=str, help='Input text string')
    parser.add_argument('--output_path', type=str,
                        help='Output file path', default='output.wav')

    # Parse the arguments
    args = parser.parse_args()
    gender = args.gender
    text = args.text
    output_path = args.output_path
    # if output_path directory does not exist, create it
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    config_file = "config.json"

    assert gender in ['male', 'female']
    if gender == 'male':
        duration_model_path = "ckpts/vbx_duration_model.pth"
        lightspeed_model_path = "ckpts/generator_male.pth"
        phone_set_file = "ckpts/vbx_phone_set.json"
    else:
        duration_model_path = "ckpts/duration_model.pth"
        lightspeed_model_path = "ckpts/generator_female.pth"
        phone_set_file = "ckpts/phone_set.json"

    # load config file
    with open(config_file, "rb") as f:
        hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    # load phone set json file
    with open(phone_set_file, "r") as f:
        phone_set = json.load(f)

    assert phone_set[0][1:-1] == "SEP"
    assert "sil" in phone_set
    sil_idx = phone_set.index("sil")
    space_re = regex.compile(r"\s+")
    number_re = regex.compile("([0-9]+)")
    digits = ["không", "một", "hai", "ba", "bốn",
              "năm", "sáu", "bảy", "tám", "chín"]
    num_re = regex.compile(r"([0-9.,]*[0-9])")
    alphabet = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵbcdđghklmnpqrstvx"
    keep_text_and_num_re = regex.compile(rf"[^\s{alphabet}.,0-9]")
    keep_text_re = regex.compile(rf"[^\s{alphabet}]")

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # infer & save result
    sampling_rate, audio = speak(text)
    sf.write(output_path, audio, sampling_rate)
