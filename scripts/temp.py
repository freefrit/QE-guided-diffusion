import sys
root = '/home/evan/231121_dmc_latent_variable_kernel'
if root not in sys.path:
    sys.path.insert(0, root)  # add ROOT to PATH

import argparse
import os

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import compressai

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import date, datetime
import math
import random
from io import BytesIO

today = datetime.now()
# img_name = 'kodim04'
# img_path = f'/dataset/kodak/{img_name}.png'
img_name = 'img_016'
img_path = '/work/240805_QE/240806_samples_20x256x256x3/img_016.png'
root = f'/work/240805_QE/{today.strftime("%y%m%d%H%M")}_{img_name}'
if not os.path.exists(root): os.mkdir(root)
cond_scale = 1
codec_q = 1
codec_metric = 'mse'
x_constraint = False
num_sample = 4
simul = False
ga_weight = 10
logger.configure(dir=root)
logger.log(f'scale:{cond_scale}, codec_q:{codec_q}, metric:{codec_metric}')
logger.log(f'x_constraint:{x_constraint}, num_sample:{num_sample}, simul:{simul}')
ga_feats_gt = []
steps = []
steps.extend(range(30, 200, 30))
steps.extend(range(200, 1000, 100))
steps.append(999)

def show_model_size(net):
    print("========= Model Size =========")
    total = 0
    for name, module in net.named_children():
        sum = 0
        for param in module.parameters():
            sum += param.numel()
        total += sum
        print(f"{name}: {sum/1e6:.3f} M params")
    print("==============================")
    print(f"Total: {total/1e6:.3f} M params\n")

def jpeg_compression(image, qf=None):
    if qf == None:
        qf = random.randrange(50, 100)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimize=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def normalize_tensor(in_feat, eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    norm_factor = torch.linalg.vector_norm(in_feat, dim=1, keepdim=True)
    return in_feat / (norm_factor + eps)

class UnNormalize(object):
    def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        temp = tensor.detach().clone()
        for t, m, s in zip(temp, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return temp

def get_ga_hook(outputs_list):
    def ga_hook(module, input, output):
        outputs_list.append(output)
    return ga_hook

class ste_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, x_hat_grad):
        return x_hat_grad.clone()

class CodecOperator():
    def __init__(self, q, x_constraint=False):
        self.codec = compressai.zoo.mbt2018_mean(quality=q, metric=codec_metric, pretrained=True, progress=True).cuda().eval()
        self.x_constraint = x_constraint
    
    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 
    
    def forward(self, data, **kwargs):
        y = self.codec.g_a((data + 1.0) / 2.0)
        if self.x_constraint:
            z = self.codec.h_a(y)
            z_hat, _ = self.codec.entropy_bottleneck(z)
            gaussian_params = self.codec.h_s(z_hat)
            _, means_hat = gaussian_params.chunk(2, 1)
            y_hat = ste_round.apply(y - means_hat)
            x_hat = self.codec.g_s(y_hat + means_hat)
            return (x_hat * 2.0) - 1.0
        
        y_hat = ste_round.apply(y)    
        return y_hat

#PosteriorSampling
class ConditioningMethod():
    def __init__(self, **kwargs):
        self.operator = kwargs.get('operator')
        self.scale = kwargs.get('scale', 1.0)
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        norm = torch.linalg.norm(difference)
        # norm = torch.linalg.vector_norm(difference, ord=2, dim=(1, 2, 3))
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        # norm_grad = torch.autograd.grad(outputs=torch.mean(norm), inputs=x_prev)[0]
        # norm_grad = torch.autograd.grad(outputs=torch.sum(norm), inputs=x_prev)[0]
        # print(norm_grad)           
        return norm_grad, norm

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, total_step, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        
        # if idx > (0.5*total_step):
        x_t -= norm_grad * self.scale
        return x_t, norm
    
class ConditioningMethod2(ConditioningMethod):
    def __init__(self, **kwargs):
        self.operator = kwargs.get('operator')
        self.scale = kwargs.get('scale', 1.0)
    # method for regularization on x0
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        norm = torch.linalg.norm(difference)
        # norm = torch.linalg.vector_norm(difference, ord=2, dim=(1, 2, 3))
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        # norm_grad = torch.autograd.grad(outputs=torch.mean(norm), inputs=x_prev)[0]
        # norm_grad = torch.autograd.grad(outputs=torch.sum(norm), inputs=x_0_hat)[0]
        # print(norm_grad)           
        return norm_grad, norm

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, total_step, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=None, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_0_hat -= norm_grad * self.scale / kwargs.get('diffusion').sqrt_alphas_cumprod[idx]
        # x_0_hat -= norm_grad * self.scale
        # if idx > (0.5*total_step):
        # x_t -= norm_grad * self.scale

        # sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        noise = th.randn_like(x_0_hat)
        t = torch.tensor([idx] * x_0_hat.shape[0], device=x_0_hat.device)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_0_hat.shape) - 1)))
        )  # no noise when t == 0
        model_mean, _, _ = kwargs.get('diffusion').q_posterior_mean_variance(
            x_start=x_0_hat, x_t=x_prev, t=t
        )
        update_x_t = model_mean + nonzero_mask * th.exp(0.5 * kwargs.get('x_t_log_var')) * noise

        return update_x_t, norm

class CodecConditioning(ConditioningMethod):
    def __init__(self, **kwargs):
        self.operator = kwargs.get('operator')
        self.scale = kwargs.get('scale', 1.0)
        # self.ga_feats_gt = kwargs.get('ga_feats_gt')
        self.ga_feats_pred = []

        for layer in self.operator.codec.g_a:
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(get_ga_hook(self.ga_feats_pred))

    def get_mean_feats(self, ga_feats):
        if isinstance(ga_feats, list):
            feats = torch.cat([torch.mean(feature, dim=(-1, -2)) for feature in ga_feats], dim=1)
        else:
            feats = torch.mean(ga_feats, dim=(-1, -2))
        feats = normalize_tensor(feats)
        return feats
    
    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, **kwargs):
        '''
        x_prev: input noisy img x_(t+1)
        x_t: diffusion model predicted step t noisy img
        x_t_mean: predicted step t mean
        x_0_hat: predicted clear img
        measurement: GT condition y for regularization
        idx: timestep t
        ga_feats_g: GT conditions of codec's g_a
        k: # for taking top k values
        '''
        self.ga_feats_pred.clear()
        # norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        norm = torch.linalg.norm(difference)
        loss = norm

        # Take topk from all feature channels
        # feats_gt = self.get_mean_feats(kwargs.get('ga_feats_gt'))
        # feats_pred = self.get_mean_feats(self.ga_feats_pred)

        # topk_idx = torch.topk(torch.abs(feats_gt - feats_pred) * feats_gt, kwargs.get('k'), dim=1)[1].squeeze()
        # feats_gt_k = feats_gt[:, topk_idx]
        # feats_pred_k = feats_pred[:, topk_idx]

        # loss = loss + ga_weight*torch.mean(torch.abs(feats_gt_k - feats_pred_k)**2)

        # Take topk from each conv output feature channels
        for i in range(len(self.ga_feats_pred)):
            feats_gt = self.get_mean_feats(kwargs.get('ga_feats_gt')[i])
            feats_pred = self.get_mean_feats(self.ga_feats_pred[i])

            topk_idx = torch.topk(torch.abs(feats_gt - feats_pred) * feats_gt, kwargs.get('k'), dim=1)[1].squeeze()
            feats_gt_k = feats_gt[:, topk_idx]
            feats_pred_k = feats_pred[:, topk_idx]

            loss = loss + ga_weight*torch.mean(torch.abs(feats_gt_k - feats_pred_k)**2)
        # print(f'basic norm:{norm.item()}, codec norm:{(loss-norm).item()}')
        ga_grad = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]

        x_t -= ga_grad * self.scale
        return x_t, norm

def p_sample_loop(model,
                  diffusion,
                  x_start,
                  step,
                  measurement,
                  measurement_cond_fn,
                  sample_idx=None,
                  ):
    img = x_start
    device = x_start.device
    unorm = UnNormalize()
    plotx, ploty = [], []
    for i in tqdm(range(step, 0, -1)):
        time = torch.tensor([i] * img.shape[0], device=device)
        img = img.requires_grad_()
        out = diffusion.p_sample(model=model, x=img, t=time)
        mean_var = diffusion.p_mean_variance(model=model, x=img, t=time)

        # noisy_measurement = diffusion.q_sample(measurement, t=time)
        img, distance = measurement_cond_fn(x_prev=img,
                                            x_t=out['sample'],
                                            x_t_mean=mean_var['mean'],
                                            x_0_hat=out['pred_xstart'],
                                            measurement=measurement,
                                            idx=i,
                                            total_step=step,
                                            x_t_log_var=mean_var['log_variance'],
                                            diffusion=diffusion,
                                            t=time,
                                            # noisy_measurement=noisy_measurement,
                                            # sigma_t=torch.exp(0.5 * mean_var['log_variance']),
                                            )
        img = img.detach_()

        if (i-1) % (step//10) == 0:
            plotx.append(i-1)
            ploty.append(distance.mean().item())
            if sample_idx:
                torchvision.utils.save_image(unorm(img[0]), f'{root}/from_{step}/x_{i-1:03d}_{sample_idx}.png')
            else:
                for idx in range(img.shape[0]):
                    torchvision.utils.save_image(unorm(img[idx]), f'{root}/from_{step}/x_{i-1:03d}_{idx}.png')
    # plt.figure(0)
    plt.clf()
    plt.xlabel('time step')
    plt.ylabel('distance (norm)')
    plt.plot(plotx, ploty, 'r-o')
    plt.grid()
    if sample_idx:
        plt.savefig(f'{root}/from_{step}/norm_plot_{sample_idx}.png')
    else:
        plt.savefig(f'{root}/from_{step}/norm_plot.png')
    return img, distance
        

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    # logger.configure(dir=root)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    # show_model_size(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = compressai.zoo.mbt2018_mean(quality=codec_q, metric=codec_metric, pretrained=True, progress=True).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    jpeg_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Lambda(lambda image : jpeg_compression(image, qf=10)),
        transforms.ToTensor(),
    ])
    norm = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    unorm = UnNormalize()

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    x_jpeg = jpeg_transform(img).unsqueeze(0).to(device)
    # print(f'{x[0].max().item()}, {x[0].min().item()}')
    # print(f'{unorm(x[0]).max().item()}, {unorm(x[0]).min().item()}')
    torchvision.utils.save_image(x[0], f'{root}/x.png')
    # torchvision.utils.save_image(x_jpeg[0], f'{root}/x_jpeg.jpeg')

    logger.log("codec processing...")
    with torch.no_grad():
        y = codec.g_a(x)
        z = codec.h_a(y)
        z_hat, z_likelihoods = codec.entropy_bottleneck(z)
        gaussian_params = codec.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = codec.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = codec.g_s(y_hat)
        bpp = sum(
            (torch.log(z_likelihoods).sum() / (-math.log(2) * 256*256),
            torch.log(y_likelihoods).sum() / (-math.log(2) * 256*256))
        )
        logger.log(f'bpp: {bpp.item()}')
        # print(f'x_hat range:{x_hat[0].max().item()}, {x_hat[0].min().item()}')
        torchvision.utils.save_image(x_hat[0], f'{root}/x_hat.png')

    x = norm(x)
    x_hat = norm(x_hat)
    # x_hat = norm(x_jpeg)
    # x_hat = x
    loss = torch.nn.MSELoss()
    mse = loss(x, x_hat)+1e-10
    # nc = x - x_hat
    # print(f'nc.max(): {nc.max().item()}')
    logger.log(f'mse: {mse.item()}')
    plotx, ploty, ploty2 = [], [], []
    
    logger.log("generating noisy imgs...")
    '''
    _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    '''
    for step in steps:
        # logger.log(f'step {step}')
        # print(f'mean scale {step}: {diffusion.sqrt_alphas_cumprod[step]}')
        # print(f'var scale {step}: {diffusion.sqrt_one_minus_alphas_cumprod[step]}')
        ratio = torch.sqrt(mse)*diffusion.sqrt_alphas_cumprod[step]/diffusion.sqrt_one_minus_alphas_cumprod[step]
        # max_ratio = nc.max().item()*diffusion.sqrt_alphas_cumprod[step]/diffusion.sqrt_one_minus_alphas_cumprod[step]
        # logger.log(f'ratio: {ratio}')
        # logger.log(f'ratio(dB): {10 * math.log((ratio**2).item(), 10)}')
        # logger.log(f'max_ratio: {max_ratio}')
        plotx.append(step)
        ploty.append(10 * math.log((ratio**2).item(), 10))
        # ploty2.append((max_ratio**2).item())

        t = torch.tensor([step]).to(device)
        x_hat_t = diffusion.q_sample(x_hat, t)
        # print(f'x_hat range:{x_hat[0].max().item()}, {x_hat[0].min().item()}')
        # print(f'x_hat_t range:{x_hat_t[0].max().item()}, {x_hat_t[0].min().item()}')
        if not os.path.exists(f'{root}/forward'): os.mkdir(f'{root}/forward')
        torchvision.utils.save_image(unorm(x_hat_t[0]), f'{root}/forward/x_hat_{step}.png')
    

    plt.xlabel('time step')
    plt.ylabel('ratio(dB)')
    plt.plot(plotx, ploty, 'r-o')
    # plt.plot(plotx, ploty2, 'b-o')
    # plt.legend(['N2N ratio', 'Max2N ratio'])
    plt.grid()
    plt.show()
    plt.savefig(f'{root}/ratio_plot.png')

    distances = []
    for step in steps:
        logger.log(f"denoising from step {step}...")
        if not os.path.exists(f'{root}/from_{step}'): os.mkdir(f'{root}/from_{step}')
        if not step == steps[-1]:
            t = torch.tensor([step]).to(device)
            x_hat_t = diffusion.q_sample(x_hat, t)
        else:
            x_hat_t = torch.randn_like(x_hat).to(device)

        operator = CodecOperator(q=codec_q, x_constraint=x_constraint)
        cond_method = ConditioningMethod(scale=cond_scale, operator=operator)
        measurement_cond_fn = cond_method.conditioning
        measurement = operator.forward(x)

        if simul == True:
            img = torch.stack([x_hat_t for _ in range(num_sample)], dim=0)
            measurements = torch.stack([measurement for _ in range(num_sample)], dim=0)
            sample, dis = p_sample_loop(model, diffusion, img, step, measurements, measurement_cond_fn)
        else:
            img = x_hat_t
            measurements = measurement
            total_dis = 0
            for sample_idx in range(num_sample):
                sample, dis = p_sample_loop(model, diffusion, img, step, measurements, measurement_cond_fn, sample_idx=sample_idx)
                total_dis += dis.item()
            distances.append(total_dis/num_sample)

        # for i in tqdm(range(step, 0, -1)):
        #     t = torch.tensor([i] * img.shape[0], device=device)
        #     with torch.no_grad():
        #         out = diffusion.p_sample(
        #             model,
        #             img,
        #             t,
        #             clip_denoised=args.clip_denoised,
        #             denoised_fn=None,
        #             cond_fn=None,
        #             model_kwargs=None,
        #         )
        #         img = out["sample"]
        #         # if not i == step:
        #         #     img, _ = idempotence(codec=codec, x_prev=img, x_t=out["sample"],
        #         #                         x_t_mean=diffusion.q_mean_variance(out["pred_xstart"],t-1)[0].detach(),
        #         #                         x_0_pred=out["pred_xstart"], x_0=x, t=i)
        #         # else:
        #         #     img = out["sample"]
        #         if (i-1) % (step//10) == 0:
        #             for idx in range(img.shape[0]):
        #                 torchvision.utils.save_image(unorm(img[idx]), f'{root}/from_{step}/x_{i-1:03d}_{idx}.png')
    # plt.figure(1)
    # plt.clf()
    # plt.xlabel('time step')
    # plt.ylabel('distance (norm)')
    # plt.plot(steps, distances, 'r-o')
    # plt.grid()
    # plt.savefig(f'{root}/last_norm_plot.png')


if __name__ == "__main__":
    main()