# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder

from tensorboardX import SummaryWriter
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *
from Network import TestModel


from pathlib import Path
import cv2
from PIL import Image
from torch.utils.data import Dataset


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target,  lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["y_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = lmbda * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, noisequant, training_stage
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    start = time.time()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        if training_stage > 1:
            s = random.randint(0, model.levels - 1)  # choose random level from [0, levels-1]
        else:
            s = 5

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d, noisequant, training_stage, s)

        out_criterion = criterion(out_net, d,  model.lmbda[s])
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 2000 == 0:
            print(
                f"Train epoch {epoch}  training_stage{training_stage}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f" \tlambda: {model.lmbda[s]} s: {s:.3f}, factor: {model.Gain.data[s].detach().cpu().numpy():0.4f}, |"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f} |'
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

    print(f"Train epoch {epoch}: Average losses:"
          f"\tLoss: {train_loss.avg:.3f} |"
          f"\tMSE loss: {train_mse_loss.avg:.3f} |"
          f"\tBpp loss: {train_bpp_loss.avg:.4f} |"
          f"\tTime (s) : {time.time()-start:.4f} |"
          )


    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg

def test_epoch(epoch, test_dataloader, model, criterion, noisequant, training_stage,):
    model.eval()
    device = next(model.parameters()).device
    print("Test training stage:{}, noise quantization:{}".format(training_stage, noisequant))

    loss_total = 0
    bpp_loss_total = 0
    mse_loss_total = 0
    with torch.no_grad():
        for s in range(model.levels):
            loss = AverageMeter()
            bpp_loss = AverageMeter()
            y_bpp_loss = AverageMeter()
            z_bpp_loss = AverageMeter()
            mse_loss = AverageMeter()
            aux_loss = AverageMeter()
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d, noisequant=noisequant, training_stage=training_stage, s=s)
                out_criterion = criterion(out_net, d, model.lmbda[s])

                aux_loss.update(model.aux_loss().item())
                bpp_loss.update(out_criterion["bpp_loss"].item())
                y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
                z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
                loss.update(out_criterion["loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())
            loss_total += loss.avg
            bpp_loss_total += bpp_loss.avg
            mse_loss_total += mse_loss.avg
            print(
                f"Test epoch {epoch}, lambda: {model.lmbda[s]}, s: {s}, factor: {model.Gain.data[s].cpu().numpy():0.4f}, training_stage {training_stage}:"
                f"\tLoss: {loss.avg:.3f} |"
                f"\tMSE loss: {mse_loss.avg:.3f} |"
                f"\tBpp loss: {bpp_loss.avg:.4f} |"
                f"\ty_Bpp loss: {y_bpp_loss.avg:.4f} |"
                f"\tz_Bpp loss: {z_bpp_loss.avg:.4f} |"
                f"\tAux loss: {aux_loss.avg:.4f}"
            )
    print(
        f"Test epoch {epoch} : Total Average losses:"
        f"\tLoss: {loss_total:.3f} |"
        f"\tMSE loss: {mse_loss_total:.3f} |"
        f"\tBpp loss: {bpp_loss_total:.4f} \n"
    )
    print()

    return loss_total, bpp_loss_total, mse_loss_total


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )

    parser.add_argument(
        "--N",
        default=128,
        type=int,
        help="Number of channels of main codec",
    )
    parser.add_argument(
        "--M",
        default=128,
        type=int,
        help="Number of channels of latent",
    )
    parser.add_argument(
        "--depth",
        type=int,
        nargs=4,
        default=(2, 4, 2, 4),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--heads",
        default=4,
        type=int,
        help="Number of heads in Multi-head Attention layer",
    )
    parser.add_argument(
        "--dim_head",
        default=96,
        type=int,
        help="Dimension of the Multi-head Attention layer",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout rate",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(384, 384),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", default=1926, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use the pretrain model to refine the models",
    )
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--training_stage', default=0, type=int, help='trainning stage')
    parser.add_argument("--stemode", default=0, type=int, help="Using ste round in the finetune stage")
    parser.add_argument('--savepath', default='./checkpoint', type=str, help='Path to save the checkpoint')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--loadFromSinglerate', default=0, type=int, help='load models from single rate')
    args = parser.parse_args(argv)
    return args





def main(argv):
    args = parse_args(argv)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed != 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print("device:{}".format((device)))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # net = getmodel(args.model, args.quality)
    net = TestModel(N=args.N, M=args.M, image_size=(args.patch_size[0], args.patch_size[1]),
                    depth=[args.depth[0], args.depth[1], args.depth[2], args.depth[3]], heads=args.heads,
                    dim_head=args.dim_head, dropout=args.dropout)
    net = net.to(device)
    if not os.path.exists(args.savepath):
        try:
            os.mkdir(args.savepath)
        except:
            os.makedirs(args.savepath)
    writer = SummaryWriter(args.savepath)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    criterion = RateDistortionLoss()

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:  # load from previous checkpoint
        if args.loadFromSinglerate:
            print("Loading previous checkpoint: ", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if "state_dict" in checkpoint:
                ckpt = checkpoint["state_dict"]
            else:
                ckpt = checkpoint
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in ckpt.items() if
                               k in model_dict.keys() and v.shape == model_dict[k].shape}
            print("pretrained_dict_diff")
            print(k for k in model_dict.keys() if k not in pretrained_dict)
            model_dict.update(pretrained_dict)
            print("pretrained_dict")
            print(pretrained_dict.keys())
            print("(model_dict")
            print(model_dict.keys())
            net.load_state_dict(model_dict)
        else:
            print("Loading: ", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            last_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if args.checkpoint and args.pretrained:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        last_epoch = 0

    training_stage = args.training_stage
    noisequant = True
    stemode = False  ##set the ste round finetune flag
    if args.stemode or training_stage>2:
        stemode = True
        noisequant = False

    for epoch in range(last_epoch, args.epochs):
        print("noisequant: {}, stemode:{}, training stage:{}".format(noisequant, stemode, training_stage))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_bpp, train_mse = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            noisequant,
            training_stage,
        )
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/mse', train_mse, epoch)
        writer.add_scalar('Train/bpp', train_bpp, epoch)

        loss, bpp, mse = test_epoch(epoch, test_dataloader, net, criterion, noisequant, training_stage, )
        writer.add_scalar('Test/loss', loss, epoch)
        writer.add_scalar('Test/mse', mse, epoch)
        writer.add_scalar('Test/bpp', bpp, epoch)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            DelfileList(args.savepath, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
 					"best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
            )
            if is_best:
                DelfileList(args.savepath, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                )


if __name__ == "__main__":
    main(sys.argv[1:])
