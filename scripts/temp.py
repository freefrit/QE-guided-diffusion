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
img_name = 'kodim04'
root = f'/work/240805_QE/{today.strftime("%y%m%d%H%M")}_{img_name}'
if not os.path.exists(root): os.mkdir(root)

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

# def idempotence(codec, x_prev, x_t, x_t_mean, x_0_pred, x_0, t):
#     unorm = UnNormalize()
#     eps = 1e-8
#     interval = 1
#     if t % interval == 0:
#         difference = codec.g_a(unorm(x_0)) - codec.g_a(unorm(x_0_pred))
#         norm = torch.linalg.norm(difference)
#         grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
#         print(f'grad: {grad}')
#         grad_norm = torch.linalg.norm(grad)

#         b, c, h, w = x_t.shape
#         r = torch.sqrt(torch.tensor(c * h * w)) * 1.
#         guidance_rate = 0.1

#         d_star = -r * grad / (grad_norm + eps)
#         d_sample = x_t - x_t_mean
#         mix_direction = d_sample + guidance_rate * (d_star - d_sample)
#         mix_direction_norm = torch.linalg.norm(mix_direction)
#         mix_step = mix_direction / (mix_direction_norm + eps) * r

#         return x_t_mean + mix_step, norm

#     else:
#         return x_t, torch.zeros(1)
class ste_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, x_hat_grad):
        return x_hat_grad.clone()

class CodecOperator():
    def __init__(self, q=4):
        self.codec = compressai.zoo.mbt2018_mean(quality=q, metric='mse', pretrained=True, progress=True).cuda().eval()
    
    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 
    
    def forward(self, data, **kwargs):
        y = self.codec.g_a((data + 1.0) / 2.0)
        y_hat = ste_round.apply(y)
        # z = self.codec.h_a(y)
        # z_hat, z_likelihoods = self.codec.entropy_bottleneck(z)
        # gaussian_params = self.codec.h_s(z_hat)
        # scales_hat, means_hat = gaussian_params.chunk(2, 1)
        # y_hat, y_likelihoods = self.codec.gaussian_conditional(y, scales_hat, means=means_hat)
        return (y_hat * 2.0) - 1.0

#PosteriorSampling
class ConditioningMethod():
    def __init__(self, **kwargs):
        self.operator = CodecOperator()
        self.scale = kwargs.get('scale', 1.0)
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]  
        # print(norm_grad)           
        return norm_grad, norm

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
    
def p_sample_loop(model,
                  diffusion,
                  x_start,
                  step,
                  measurement,
                  measurement_cond_fn,
                  ):
    img = x_start
    device = x_start.device
    unorm = UnNormalize()
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
                                            # noisy_measurement=noisy_measurement,
                                            # sigma_t=torch.exp(0.5 * mean_var['log_variance']),
                                            )
        img = img.detach_()

        if (i-1) % (step//10) == 0:
            for idx in range(img.shape[0]):
                torchvision.utils.save_image(unorm(img[idx]), f'{root}/from_{step}/x_{i-1:03d}_{idx}.png')

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
    logger.configure(dir=root)

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
    codec = compressai.zoo.mbt2018_mean(quality=4, metric='mse', pretrained=True, progress=True).to(device).eval()
    # show_model_size(codec)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    jpeg_transform = transforms.Compose([
        transforms.Lambda(lambda image : jpeg_compression(image, qf=10)),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    norm = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    unorm = UnNormalize()

    img = Image.open(f'/dataset/kodak/{img_name}.png').convert("RGB")
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
    
    steps = []
    steps.extend(range(10, 200, 10))
    steps.extend(range(200, 1000, 100))
    steps.append(999)
    logger.log("generating noisy imgs...")
    '''
    _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    '''
    for step in steps:
        logger.log(f'step {step}')
        # print(f'mean scale {step}: {diffusion.sqrt_alphas_cumprod[step]}')
        # print(f'var scale {step}: {diffusion.sqrt_one_minus_alphas_cumprod[step]}')
        ratio = torch.sqrt(mse)*diffusion.sqrt_alphas_cumprod[step]/diffusion.sqrt_one_minus_alphas_cumprod[step]
        # max_ratio = nc.max().item()*diffusion.sqrt_alphas_cumprod[step]/diffusion.sqrt_one_minus_alphas_cumprod[step]
        logger.log(f'ratio: {ratio}')
        logger.log(f'ratio(dB): {10 * math.log((ratio**2).item(), 10)}')
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

    for step in steps:
        logger.log(f"denoising from step {step}...")
        if not os.path.exists(f'{root}/from_{step}'): os.mkdir(f'{root}/from_{step}')
        t = torch.tensor([step]).to(device)
        x_hat_t = diffusion.q_sample(x_hat, t)
        img = torch.cat((x_hat_t, x_hat_t, x_hat_t, x_hat_t), 0)

        cond_method = ConditioningMethod(scale = 1.0)
        measurement_cond_fn = cond_method.conditioning
        operator = CodecOperator()
        measurement = operator.forward(x)
        measurements = torch.cat((measurement, measurement, measurement, measurement), 0)
        sample, dis = p_sample_loop(model, diffusion, img, step, measurements, measurement_cond_fn)

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




if __name__ == "__main__":
    main()