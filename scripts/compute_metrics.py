import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pytorch_msssim import ssim
import lpips
import torch
import matplotlib.pyplot as plt
import pyiqa
device = torch.device("cuda")

def image_to_tensor(image):
    if isinstance(image, Image.Image):
        # PIL Image
        transform = transforms.ToTensor()
        tensor_image = transform(image).unsqueeze(0)
    elif isinstance(image, np.ndarray):
        # OpenCV Image (numpy array)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_image = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        raise TypeError("Unsupported image type.")
    return tensor_image

def calculate_psnr(img1, img2):
    # img1 = torch.from_numpy(img1).float()
    # img2 = torch.from_numpy(img2).float()
    img1 = image_to_tensor(img1)
    img2 = image_to_tensor(img2)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    # img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = image_to_tensor(img1)
    img2 = image_to_tensor(img2)
    return ssim(img1, img2, data_range=1, size_average=False).item()

lpips_model = lpips.LPIPS(net='vgg')
def calculate_lpips(img1, img2):
    # img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = image_to_tensor(img1)
    img2 = image_to_tensor(img2)
    return lpips_model(img1, img2).item()

dists_metric = pyiqa.create_metric('dists', device=device)
def calculate_dists(img1, img2):
    img1 = image_to_tensor(img1)
    img2 = image_to_tensor(img2)
    img, ref = img2, img1
    return dists_metric(img, ref).item()

clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
def calculate_clipiqa(img):
    img = image_to_tensor(img)
    return clipiqa_metric(img).item()

liqe_metric = pyiqa.create_metric('liqe', device=device)
def calculate_liqe(img):
    img = image_to_tensor(img)
    return liqe_metric(img).item()

niqe_metric = pyiqa.create_metric('niqe', device=device)
def calculate_niqe(img):
    img = image_to_tensor(img)
    return niqe_metric(img).item()

def compare_images(folder, target, psnrs=None, ssims=None, lpipss=None, distss=None,
                   clipiqas=None, liqes=None, niqes=None):
    psnr_values = []
    ssim_values = []
    lpips_values = []
    dists_values = []
    clipiqa_values = []
    liqe_values = []
    niqe_values = []
    # img1 = Image.open(target)
    img1 = cv2.imread(target)
    for filename in sorted(os.listdir(folder)):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")) and '000' in filename:
            file = os.path.join(folder, filename)
            # img2 = Image.open(file)
            img2 = cv2.imread(file)

            if img2.size == img1.size:
                psnr_value = calculate_psnr(img1, img2)
                psnr_values.append(psnr_value)
                # print(f"PSNR between {filename}: {psnr_value:.4f}")
                ssim_value = calculate_ssim(img1, img2)
                ssim_values.append(ssim_value)
                # print(f"SSIM between {filename}: {ssim_value:.4f}")
                lpips_value = calculate_lpips(img1, img2)
                lpips_values.append(lpips_value)
                # print(f"LPIPS between {filename}: {lpips_value:.4f}")
                dists_value = calculate_dists(img1, img2)
                dists_values.append(dists_value)
                # print(f"DISTS between {filename}: {dists_value:.4f}")

                clipiqa_value = calculate_clipiqa(img2)
                clipiqa_values.append(clipiqa_value)
                # print(f"CLIPIQA of {filename}: {clipiqa_value:.4f}")
                liqe_value = calculate_liqe(img2)
                liqe_values.append(liqe_value)
                # print(f"LIQE of {filename}: {liqe_value:.4f}")
                niqe_value = calculate_niqe(img2)
                niqe_values.append(niqe_value)
                # print(f"NIQE of {filename}: {niqe_value:.4f}")
            else:
                print(f"Image shapes do not match for {filename}.")

    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Average PSNR: {average_psnr:.4f}")
        if not psnrs == None: psnrs.append(average_psnr)
        average_ssim = sum(ssim_values) / len(ssim_values)
        print(f"Average SSIM: {average_ssim:.4f}")
        if not ssims == None: ssims.append(average_ssim)
        average_lpips = sum(lpips_values) / len(lpips_values)
        print(f"Average LPIPS: {average_lpips:.4f}")
        if not lpipss == None: lpipss.append(average_lpips)
        average_dists = sum(dists_values) / len(dists_values)
        print(f"Average DISTS: {average_dists:.4f}")
        if not distss == None: distss.append(average_dists)
        average_clipiqa = sum(clipiqa_values) / len(clipiqa_values)
        print(f"Average CLIPIQA: {average_clipiqa:.4f}")
        if not clipiqas == None: clipiqas.append(average_clipiqa)
        average_liqe = sum(liqe_values) / len(liqe_values)
        print(f"Average LIQE: {average_liqe:.4f}")
        if not liqes == None: liqes.append(average_liqe)
        average_niqe = sum(niqe_values) / len(niqe_values)
        print(f"Average NIQE: {average_niqe:.4f}")
        if not niqes == None: niqes.append(average_niqe)
    else:
        print("No matching images found to compare.")

root = '/work/240805_QE/2408261037_img_016_mbtQ1_Qidem1'
folder = f'{root}/from_'
target = f'{root}/x.png'
plotx, psnrs, ssims, lpipss = [], [], [], []
distss, clipiqas, liqes, niqes = [], [], [], []

img1 = cv2.imread(target)
# img1 = Image.open(target)
try:
    img2 = cv2.imread(f'{root}/x_hat.png')
    # img2 = Image.open(f'{root}/x_hat.png')
except:
    img2 = cv2.imread(f'{root}/x_jpeg.jpeg')
    # img2 = Image.open(f'{root}/x_jpeg.jpeg')

if img1.size == img2.size:
    psnr_value = calculate_psnr(img1, img2)
    print(f"PSNR between x_hat: {psnr_value:.4f}")
    ssim_value = calculate_ssim(img1, img2)
    print(f"SSIM between x_hat: {ssim_value:.4f}")
    lpips_value = calculate_lpips(img1, img2)
    print(f"LPIPS between x_hat: {lpips_value:.4f}")
    dists_value = calculate_dists(img1, img2)
    print(f"DISTS between x_hat: {dists_value:.4f}")
    clipiqa_value = calculate_clipiqa(img2)
    print(f"CLIPIQA of x_hat: {clipiqa_value:.4f}")
    liqe_value = calculate_liqe(img2)
    print(f"LIQE of x_hat: {liqe_value:.4f}")
    niqe_value = calculate_niqe(img2)
    print(f"NIQE of x_hat: {niqe_value:.4f}")
plotx.append(0)
psnrs.append(psnr_value)
ssims.append(ssim_value)
lpipss.append(lpips_value)
distss.append(dists_value)
clipiqas.append(clipiqa_value)
liqes.append(liqe_value)
niqes.append(niqe_value)

steps = []
steps.extend(range(10, 200, 10))
steps.extend(range(200, 1000, 100))
steps.append(999)
# for step in range(100, 501, 100):
for step in steps:
    print(f"step: {step}")
    try:
        compare_images(folder + str(step), target, 
                       psnrs, ssims, lpipss, distss, clipiqas, liqes, niqes)
        plotx.append(step)
    except:
        break

plt.figure(num=0, figsize=(10,15))
plt.subplot(4, 1, 1)
plt.xlabel('time step')
plt.title('PSNR↑')
plt.plot(plotx, psnrs, '-o')
plt.grid()
plt.subplot(4, 1, 2)
plt.xlabel('time step')
plt.title('SSIM↑')
plt.plot(plotx, ssims, '-o')
plt.grid()
plt.subplot(4, 1, 3)
plt.xlabel('time step')
plt.title('LPIPS↓')
plt.plot(plotx, lpipss, '-o')
plt.grid()
plt.subplot(4, 1, 4)
plt.xlabel('time step')
plt.title('DISTS↓' if dists_metric.lower_better else 'DISTS↑')
plt.plot(plotx, distss, '-o')
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig(f'{root}/metrics_FR.png')

plt.figure(num=1, figsize=(10,10))
plt.subplot(3, 1, 1)
plt.xlabel('time step')
plt.title('CLIPIQA↓' if clipiqa_metric.lower_better else 'CLIPIQA↑')
plt.plot(plotx, clipiqas, '-o')
plt.grid()
plt.subplot(3, 1, 2)
plt.xlabel('time step')
plt.title('LIQE↓' if liqe_metric.lower_better else 'LIQE↑')
plt.plot(plotx, liqes, '-o')
plt.grid()
plt.subplot(3, 1, 3)
plt.xlabel('time step')
plt.title('NIQE↓' if niqe_metric.lower_better else 'NIQE↑')
plt.plot(plotx, niqes, '-o')
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig(f'{root}/metrics_NR.png')

# img1 = cv2.imread('/work/240805_QE/2408140816_kodim04_jpeg010/x_jpeg.jpeg')
# img2 = cv2.imread('/work/240805_QE/2408140816_kodim04_jpeg010/compressed_image.jpg')
# print(calculate_psnr(img1, img2))