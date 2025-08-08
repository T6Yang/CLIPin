import os
import logging
import time
from time import gmtime, strftime

import math
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from clip.model import CLIP
from logger import setup_primary_logging, setup_worker_logging

from data import get_data
from scheduler import cosine_lr
import collections.abc
from itertools import repeat

from downstream import downstreamtest


NAME = "debug"
GPU = 3
DEBUG = False
PRECISION = "amp"
GRAD_CKPT = True
MAX_TRAIN_EPOCHS = 50
VALID_EPOCH_INTERVAL = 1
SAVE_EPOCH_FREQUENCY = 1
LR = 3e-5
WARMUP = 100
USE_SEP_AUG = True
ONLYCLIP = False
TESTMODE = False
DATASET_NAME = "COCO"
RESUME = None
RESET_OPTIMIZER = False
SHOULD_SAVE = True
DOWNSTREAM_DATASET = "PASCAL"   # ["PASCAL", "CIFAR10", "CIFAR100", "SUN397", "OxfordPets", "Caltech101"]
DOWNSTREAM_TASK = "LP"  # ["LP", "FFT", "ZSC"] # Linear Probing, Fully Fine-tune, Zero-shot classification

BASE_PATH = "/path/to/your/experiments/{}".format(NAME)
TIME_SUFFIX = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
LOG_PATH = "{}/out_{}.log".format(BASE_PATH, TIME_SUFFIX)
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
CKPT_PATH = "{}/checkpoints".format(BASE_PATH)
DEVICE = "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"


if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH, exist_ok=True)

log_queue = setup_primary_logging(LOG_PATH, LOG_LEVEL)
setup_worker_logging(log_queue, LOG_LEVEL)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

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

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)

    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def get_loss(model, images1, images2, texts, use_sep_aug, onlyclip):
    image_features, image_features_predicted, image_features_predicted_self, image_features_contrast, text_features, text_features_predicted, text_features_predicted_self, text_features_contrast, logit_scale, lambda_selfalign, lambda_combine = model(images1, images2, texts, use_sep_aug, DEVICE)

    logit_scale = logit_scale.mean()

    logits_per_image = logit_scale * image_features_contrast @ text_features_contrast.t()
    logits_per_text = logits_per_image.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.to(DEVICE, non_blocking=True)

    CLIP_Loss = (
                        nn.functional.cross_entropy(logits_per_image, ground_truth)
                        + nn.functional.cross_entropy(logits_per_text, ground_truth)
                ) / 2

    BYOL_Loss = 2 - (
                        cosine_similarity(image_features_predicted, text_features).mean()
                        + cosine_similarity(text_features_predicted, image_features).mean()
                )

    selfalign_Loss = 2 - (
                        cosine_similarity(image_features_predicted_self, image_features).mean()
                        + cosine_similarity(text_features_predicted_self, text_features).mean()
                )

    if onlyclip:
        logits_per_image_acc = logit_scale * image_features_contrast @ text_features_contrast.t()
    else:
        logits_per_image_acc = logit_scale * image_features @ text_features_predicted.t()
    logits_per_text_acc = logits_per_image_acc.t()

    i2t_acc = (logits_per_image_acc.argmax(-1) == ground_truth).sum() / len(logits_per_image_acc)
    t2i_acc = (logits_per_text_acc.argmax(-1) == ground_truth).sum() / len(logits_per_text_acc)
    acc = {"i2t": i2t_acc, "t2i": t2i_acc}

    if onlyclip:
        total_loss = 0 * BYOL_Loss + 0 * lambda_selfalign * selfalign_Loss + lambda_combine * CLIP_Loss
    else:
        total_loss = BYOL_Loss + lambda_selfalign * selfalign_Loss + lambda_combine * CLIP_Loss

    return total_loss, acc, BYOL_Loss, selfalign_Loss, CLIP_Loss


def train(model, dataloader, epoch, optimizer, scaler, scheduler, global_trained_steps, use_sep_aug=False, onlyclip=False):

    model.train()
    num_batches_per_epoch = len(dataloader)
    data_iter = iter(dataloader)

    end = time.time()
    epoch_trained_steps = 0
    num_samples = 0
    for i in range(global_trained_steps - num_batches_per_epoch * epoch, num_batches_per_epoch):
        batch = next(data_iter)
        step = num_batches_per_epoch * epoch + i

        optimizer.zero_grad()
        scheduler(step)

        images1, images2, texts, _, _ = batch
        images1 = images1.to(DEVICE, non_blocking=True)
        images2 = images2.to(DEVICE, non_blocking=True)
        if not isinstance(texts, list):
            texts = texts.to(DEVICE, non_blocking=True)

        data_time = time.time() - end

        if PRECISION == "amp":
            with autocast():
                total_loss, acc, BYOL_Loss, selfalign_Loss, CLIP_Loss = get_loss(model, images1, images2, texts, use_sep_aug, onlyclip)
                scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            total_loss, acc, BYOL_Loss, selfalign_Loss, CLIP_Loss = get_loss(model, images1, images2, texts, use_sep_aug, onlyclip)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
        model.lambda_selfalign.data = torch.clamp(model.lambda_selfalign.data, 0, 20)
        model.lambda_combine.data = torch.clamp(model.lambda_combine.data, 0, 20)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        num_samples += len(images1)
        samples_per_epoch = len(dataloader.dataset)
        percent_complete = 100.0 * num_samples / samples_per_epoch

        logging.info(
            f"Global Steps: {step + 1}/{num_batches_per_epoch * MAX_TRAIN_EPOCHS} | " +
            f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
            f"Total Loss: {total_loss.item():.6f} | " +
            f"Loss_BYOL: {BYOL_Loss.item():.6f} | " +
            f"Loss_selfalign: {selfalign_Loss.item():.6f} | " +
            f"Loss_CLIP: {CLIP_Loss.item():.6f} | " +
            f"I2T Acc: {acc['i2t'].item() * 100:.2f} | " +
            f"T2I Acc: {acc['t2i'].item() * 100:.2f} | " +
            f"Data Time: {data_time:.3f}s | " +
            f"Batch Time: {batch_time:.3f}s | " +
            f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
            f"logit_scale: {model.logit_scale.data.exp():.3f} | " +
            f"lambda_selfalign: {model.lambda_selfalign.data:.3f} | " +
            f"lambda_combine: {model.lambda_combine.data:.3f} | " +
            f"Global Batch Size: {len(images1)}"
        )

    return epoch_trained_steps


def evaluate(model, dataloader, epoch, global_trained_steps, onlyclip=True):
    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, global_trained_steps))

    if DOWNSTREAM_TASK == "LP":
        epoch_main, auc, map, loss_cls = downstreamtest(model, DOWNSTREAM_DATASET, DOWNSTREAM_TASK, epoch + 1, DEVICE, onlyclip)
        logging.info(
            f"MultiLabelCls Validation Result (epoch {epoch_main}) | "
            f"Valid Loss: {loss_cls:.6f} | "
            f"Valid AUC: {auc:.6f} | "
            f"Valid_MAP: {map:.6f}"
        )

    elif DOWNSTREAM_TASK == "FFT":
        epoch_main, auc, map, loss_cls = downstreamtest(model, DOWNSTREAM_DATASET, DOWNSTREAM_TASK, epoch + 1, DEVICE,
                                                        onlyclip)
        logging.info(
            f"MultiLabelCls Validation Result (epoch {epoch_main}) | "
            f"Valid Loss: {loss_cls:.6f} | "
            f"Valid AUC: {auc:.6f} | "
            f"Valid_MAP: {map:.6f}"
        )

    elif DOWNSTREAM_TASK == "ZSC":
        for downstream_dataset in DOWNSTREAM_DATASET:
            epoch_main, acc, auc, map = downstreamtest(model, downstream_dataset, DOWNSTREAM_TASK, epoch + 1, DEVICE,
                                                            onlyclip)
            logging.info(
                f"Zero-shot Classification Result (epoch {epoch_main}) | "
                f"Zero-shot Classification Accuracy: {acc} |"
                f"Valid AUC: {auc:.6f} | "
                f"Valid_MAP: {map:.6f}"
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

    if PRECISION == "amp" or PRECISION == "fp32":
        convert_models_to_fp32(model)

    if GRAD_CKPT:
        model.set_grad_checkpointing()
        logging.info("Grad-checkpointing activated.")

    model = model.to(DEVICE)

    train_dataloader = get_data(dataset_name=DATASET_NAME, split="train", batch_size=256, num_workers=4)
    val_dataloader = get_data(dataset_name=DATASET_NAME, split="val", batch_size=256, num_workers=4)

    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.001},
        ],
        lr=LR,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )
    total_steps = len(train_dataloader) * MAX_TRAIN_EPOCHS
    scheduler = cosine_lr(optimizer, LR, WARMUP, total_steps)
    scaler = GradScaler() if PRECISION == "amp" else None

    if RESUME is not None:
        if os.path.isfile(RESUME):
            logging.info(
                f"=> begin to load checkpoint '{RESUME}'"
            )
            checkpoint = torch.load(RESUME, map_location="cpu")

            # # load self-trained CLIP
            # sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
            # model.load_state_dict(sd, False)

            # load original CLIP
            sd = {k: v for k, v in checkpoint.state_dict().items() if "bert.pooler" not in k}
            sd_transformer = {k: v for k, v in checkpoint.state_dict().items() if k.startswith('transformer.')}
            sd_ln_final = {k: v for k, v in checkpoint.state_dict().items() if "ln_final" in k}
            sd_token_embedding = {k: v for k, v in checkpoint.state_dict().items() if "token_embedding" in k}
            sd_positional_embedding = {k: v for k, v in checkpoint.state_dict().items() if "positional_embedding" in k}
            resize_pos_embed(sd, model)
            model.load_state_dict(sd, False)
            model.TextEncoder.load_state_dict(sd_transformer, False)
            model.TextEncoder.load_state_dict(sd_ln_final, False)
            model.TextEncoder.load_state_dict(sd_token_embedding, False)
            model.TextEncoder.load_state_dict(sd_positional_embedding, False)

            # # if load original CLIP, ban below
            # # if not RESET_OPTIMIZER and optimizer is not None:
            # #     steps = checkpoint["step"]
            # #     optimizer.load_state_dict(checkpoint["optimizer"])
            # #     logging.info("=> optimizer state is restored from the checkpoint")
            # logging.info(
            #     f"=> loaded checkpoint '{RESUME}' (epoch {checkpoint['epoch']})"
            # )
            # and unban below
            model.logit_scale.data.fill_(2.659260036932778)
            logging.info(
                f"=> loaded checkpoint '{RESUME}'"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(RESUME))

    logging.info("Params:")
    params_file = os.path.join(BASE_PATH, "params_{}.txt".format(TIME_SUFFIX))
    with open(params_file, "w", encoding="utf-8") as f:
        f.write(f"name: {NAME}\n")
        logging.info(f"name: {NAME}")
        f.write(f"gpu_device: {GPU}\n")
        logging.info(f"gpu_device: {GPU}")
        f.write(f"debug: {DEBUG}\n")
        logging.info(f"debug: {DEBUG}")
        f.write(f"precision: {PRECISION}\n")
        logging.info(f"precision: {PRECISION}")
        f.write(f"gradient_checkpointing: {GRAD_CKPT}\n")
        logging.info(f"gradient_checkpointing: {GRAD_CKPT}")
        f.write(f"max_training_epochs: {MAX_TRAIN_EPOCHS}\n")
        logging.info(f"max_training_epochs: {MAX_TRAIN_EPOCHS}")
        f.write(f"valid_epoch_interval: {VALID_EPOCH_INTERVAL}\n")
        logging.info(f"valid_epoch_interval: {VALID_EPOCH_INTERVAL}")
        f.write(f"save_epoch_frequency: {SAVE_EPOCH_FREQUENCY}\n")
        logging.info(f"save_epoch_frequency: {SAVE_EPOCH_FREQUENCY}")
        f.write(f"learning_rate: {LR}\n")
        logging.info(f"learning_rate: {LR}")
        f.write(f"warmup: {WARMUP}\n")
        logging.info(f"warmup: {WARMUP}")
        f.write(f"use_seperated_augmentations: {USE_SEP_AUG}\n")
        logging.info(f"use_seperated_augmentation: {USE_SEP_AUG}")
        f.write(f"only_clip: {ONLYCLIP}\n")
        logging.info(f"only_clip: {ONLYCLIP}")
        f.write(f"test_mode: {TESTMODE}\n")
        logging.info(f"test_mode: {TESTMODE}")
        f.write(f"dataset_name: {DATASET_NAME}\n")
        logging.info(f"dataset_name: {DATASET_NAME}")
        f.write(f"resume: {RESUME}\n")
        logging.info(f"resume: {RESUME}")
        f.write(f"reset_optimizer: {RESET_OPTIMIZER}\n")
        logging.info(f"reset_optimizer: {RESET_OPTIMIZER}")
        f.write(f"should_save: {SHOULD_SAVE}\n")
        logging.info(f"should_save: {SHOULD_SAVE}")
        f.write(f"downstream_dataset: {DOWNSTREAM_DATASET}\n")
        logging.info(f"downstream_dataset: {DOWNSTREAM_DATASET}")
        f.write(f"downstream_task: {DOWNSTREAM_TASK}\n")
        logging.info(f"downstream_task: {DOWNSTREAM_TASK}")


    if DEVICE != 'cpu':
        logging.info(f"Use GPU: {DEVICE[-1]} for training")


    start_epoch = 0
    steps = 0

    for epoch in range(start_epoch, MAX_TRAIN_EPOCHS):
        logging.info(f'Start epoch {epoch + 1}')
        for param in model.parameters():
            param.requires_grad = True

        if TESTMODE:
            steps = -1

        else:
            num_steps_this_epoch = train(model, train_dataloader, epoch, optimizer, scaler, scheduler, steps, USE_SEP_AUG, ONLYCLIP)
            steps += num_steps_this_epoch

            # Saving checkpoints.
            if SHOULD_SAVE and num_steps_this_epoch > 0:
                if (epoch + 1) == MAX_TRAIN_EPOCHS or (
                        SAVE_EPOCH_FREQUENCY > 0 and ((epoch + 1) % SAVE_EPOCH_FREQUENCY) == 0
                ):
                    t1 = time.time()
                    save_path = os.path.join(CKPT_PATH, f"epoch{epoch + 1}.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "step": steps,
                            "name": NAME,
                            "state_dict": model.state_dict(),
                            # "optimizer": optimizer.state_dict(),
                        },
                        save_path,
                    )
                    logging.info(
                        "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                                     steps,
                                                                                                     time.time() - t1))

        if (epoch + 1) % VALID_EPOCH_INTERVAL == 0:
            evaluate(model, val_dataloader, epoch, steps)

        # if exists next epoch, reload the dataset and dataloader for the next epoch
        if epoch + 1 < MAX_TRAIN_EPOCHS:
            train_dataloader = get_data(dataset_name=DATASET_NAME, split="train", batch_size=256, num_workers=4)


if __name__ == "__main__":
    main()