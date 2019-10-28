import argparse
import os
import time

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from dataset import get_loader
from util import (eval_forward, evaluate_scores, get_models,
                  save_numpy_array_as_image, set_eval)


def save_codes(name, codes):
    print(codes)
    codes = (codes.astype(np.int8) + 1) // 2
    export = np.packbits(codes.reshape(-1))
    np.savez_compressed(
        name + '.codes',
        shape=codes.shape,
        codes=export)


def save_output_images(name, ex_imgs):
    for i, img in enumerate(ex_imgs):
        save_numpy_array_as_image(
            '%s_iter%02d.png' % (name, i + 1),
            img
        )


def finish_batch(args, filenames, original, out_imgs,
                 losses, code_batch, output_suffix):

    all_losses, all_msssim, all_psnr = [], [], []
    for ex_idx, filename in enumerate(filenames):
        filename = filename.split('/')[-1]
        if args.save_codes:
            save_codes(
                os.path.join(args.out_dir, output_suffix, 'codes', filename),
                code_batch[:, ex_idx, :, :, :]
            )

        if args.save_out_img:
            save_output_images(
                os.path.join(args.out_dir, output_suffix, 'images', filename),
                out_imgs[:, ex_idx, :, :, :]
            )

        msssim, psnr = evaluate_scores(
            original[None, ex_idx],
            [out_img[None, ex_idx] for out_img in out_imgs])

        all_losses.append(losses)
        all_msssim.append(msssim)
        all_psnr.append(psnr)

    return all_losses, all_msssim, all_psnr


def run_eval(model, eval_loader, args, output_suffix=''):
    with torch.no_grad():
        for sub_dir in ['codes', 'images']:
            cur_eval_dir = os.path.join(args.out_dir, output_suffix, sub_dir)
            if not os.path.exists(cur_eval_dir):
                print("Creating directory %s." % cur_eval_dir)
                os.makedirs(cur_eval_dir)

        all_losses, all_msssim, all_psnr = [], [], []
        total_baseline_scores = np.array([0., 0.])

        start_time = time.time()
        for i, (batch, ctx_frames, filenames) in enumerate(eval_loader):
            f1, f2 = batch[:, :3].numpy(), batch[:, 6:9].numpy()
            batch = batch.cuda()

            original, out_imgs, losses, code_batch, baseline_scores = eval_forward(
                model, (batch, ctx_frames), args)

            print(np.mean(f1), np.median(f1), np.linalg.norm(f1),
                  np.mean(f2), np.median(f2), np.linalg.norm(f2))

            losses, msssim, psnr = finish_batch(
                args, filenames, original, out_imgs,
                losses, code_batch, output_suffix)

            all_losses += losses
            all_msssim += msssim
            all_psnr += psnr

            if i % 10 == 0:
                print('\tevaluating iter %d (%f seconds)...' % (
                    i, time.time() - start_time))

            total_baseline_scores += baseline_scores

        return (np.array(all_losses).mean(axis=0),
                np.array(all_msssim).mean(axis=0),
                np.array(all_psnr).mean(axis=0),
                total_baseline_scores / len(eval_loader))
