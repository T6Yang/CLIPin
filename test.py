import os
import logging
from time import gmtime, strftime

import math
import torch
import torch.nn.functional as F
from clip.model import CLIP
from logger import setup_primary_logging, setup_worker_logging

import collections.abc
from itertools import repeat

from downstream import downstreamtest


NAME = "testZSC"
GPU = 1
DEBUG = False
RESUME = [
    "/path/to/your/ckpt"
]
# DOWNSTREAM_DATASET = "PASCAL"   # ["PASCAL", "CIFAR10", "CIFAR100", "SUN397", "Caltech101"]
DOWNSTREAM_DATASET = ["PASCAL"]
DOWNSTREAM_TASK = "ZSC"  # ["LP", "ZSC"]
ONLYCLIP = True

BASE_PATH = "/path/to/your/experiments/{}".format(NAME)
TIME_SUFFIX = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
LOG_PATH = "{}/out_{}.log".format(BASE_PATH, TIME_SUFFIX)
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
DEVICE = "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH, exist_ok=True)


log_queue = setup_primary_logging(LOG_PATH, LOG_LEVEL)
setup_worker_logging(log_queue, LOG_LEVEL)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1, prefix=""):
    to_2tuple = _ntuple(2)
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get(prefix + 'visual.positional_embedding', None)
    model = model.module if hasattr(model, 'module') else model
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict[prefix + 'visual.positional_embedding'] = new_pos_embed


def evaluate(model, epoch, global_trained_steps, ONLYCLIP):
    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, global_trained_steps))

    for downstream_dataset in DOWNSTREAM_DATASET:
        epoch_main, acc, auc, map = downstreamtest(model, downstream_dataset, DOWNSTREAM_TASK, epoch + 1, DEVICE, ONLYCLIP)
        logging.info(
            f"MultiLabelCls Validation Result (epoch {epoch_main}) | "
            f"Valid acc: {acc} | "
            f"Valid AUC: {auc} | "
            f"Valid_(m)AP: {map} | "
        )


def main():

    model = CLIP(embed_dim=512,
                 image_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=16,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12
    )   # initialized ViT-B/16

    model = model.to(DEVICE)

    logging.info("Params:")
    params_file = os.path.join(BASE_PATH, "params_{}.txt".format(TIME_SUFFIX))
    with open(params_file, "w", encoding="utf-8") as f:
        f.write(f"name: {NAME}\n")
        logging.info(f"name: {NAME}")
        f.write(f"gpu_device: {GPU}\n")
        logging.info(f"gpu_device: {GPU}")
        f.write(f"debug: {DEBUG}\n")
        logging.info(f"debug: {DEBUG}")
        f.write(f"downstream_dataset: {DOWNSTREAM_DATASET}\n")
        logging.info(f"downstream_dataset: {DOWNSTREAM_DATASET}")
        f.write(f"downstream_task: {DOWNSTREAM_TASK}\n")
        logging.info(f"downstream_task: {DOWNSTREAM_TASK}")
        f.write(f"resume: {RESUME}\n")
        logging.info(f"resume: {RESUME}")

    if DEVICE != 'cpu':
        logging.info(f"Use GPU: {DEVICE[-1]} for training")

    if RESUME is not None:
        if isinstance(RESUME, str):
            if os.path.isfile(RESUME):
                logging.info(
                    f"=> begin to load checkpoint '{RESUME}'"
                )

                checkpoint = torch.load(RESUME, map_location="cpu")

                # load self-trained CLIP
                sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
                resize_pos_embed(sd, model)
                model.load_state_dict(sd, False)

                logging.info(
                    f"=> loaded checkpoint '{RESUME}' (epoch {checkpoint['epoch']})"
                )

                evaluate(model, -1, -1, ONLYCLIP)
            else:
                logging.info("=> no checkpoint found at '{}'".format(RESUME))
        if isinstance(RESUME, list):
            for resume in RESUME:
                if os.path.isfile(resume):
                    logging.info(
                        f"=> begin to load checkpoint '{resume}'"
                    )

                    checkpoint = torch.load(resume, map_location="cpu")

                    # load self-trained CLIP
                    sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
                    resize_pos_embed(sd, model)
                    model.load_state_dict(sd, False)

                    logging.info(
                        f"=> loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})"
                    )

                    evaluate(model, -1, -1, ONLYCLIP)
                else:
                    logging.info("=> no checkpoint found at '{}'".format(resume))


if __name__ == "__main__":
    main()