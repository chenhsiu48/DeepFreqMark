#!/usr/bin/env -S uv run

import argparse
import os
import glob
import random
import torch
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
import numpy as np
from collections import defaultdict
from colorama import Style, Fore
import datetime
import time
import torch.nn.functional as F
from chlib.common import *
import sys
import torch_dct as dct
import torchvision
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_tensor
import pandas as pd

WM_MESSAGE_LENGTH = 32

LDM_MODELS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd21": "Manojb/stable-diffusion-2-1-base",
    #"sdt": "stabilityai/sd-turbo",
    "opj": "prompthero/openjourney",
}

def load_prompts_from_file(file_path):
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def calculate_psnr(img1, img2):
    t1 = transforms.ToTensor()(img1)
    t2 = transforms.ToTensor()(img2)
    mse = torch.mean((t1 - t2) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()

class ConvBNLeakyRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, last_block=False):
        super().__init__()
        layers = [
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
        ]
        if last_block == False:
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)

class DeConvBNLeakyRelu(nn.Module):
    def __init__(self, channels_in, channels_out, last_block=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(channels_in, channels_out, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channels_out),
        ]
        if last_block == False:
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)

class MessageEncoder(nn.Module):
    def __init__(self, message_length, wm_channels=1, filters=64, wm_dim=32):
        super().__init__()
        self.message_length = message_length
        self.wm_dim = wm_dim
        self.wm_channels = wm_channels
        self.base = self.wm_dim * self.wm_dim
        self.mlp = nn.Linear(message_length, self.base)
        self.convs = nn.Sequential(*[
            ConvBNLeakyRelu(1, filters), 
            ConvBNLeakyRelu(filters, filters), 
            ConvBNLeakyRelu(filters, self.wm_channels, last_block=True),
        ])
        
    def forward(self, x):
        h = self.mlp(x) # [B, base]
        h = h.view(-1, 1, self.wm_dim, self.wm_dim)
        return self.convs(h)

class MessageDecoder(nn.Module):
    def __init__(self, message_length, wm_channels=1, filters=64, wm_dim=32):
        super().__init__()
        self.message_length = message_length
        self.wm_dim = wm_dim
        self.base = self.wm_dim * self.wm_dim
        self.wm_channels = wm_channels

        self.convs = nn.Sequential(*[
            ConvBNLeakyRelu(self.wm_channels, filters),
            ConvBNLeakyRelu(filters, filters),
            ConvBNLeakyRelu(filters, 1, last_block=True),
        ])
        self.mlp = nn.Linear(self.base, message_length)

    def forward(self, x):
        h = self.convs(x)
        h = self.mlp(h.view(h.size(0), -1))
        return h

class WMModel(nn.Module):
    MODEL_BASE_NAME = 'wmmodel_epoch'
    
    def __init__(self, message_length, wm_channels=1, wm_dim=32):
        super().__init__()
        self.message_length = message_length
        self.wm_dim = wm_dim
        self.wm_channels = wm_channels

        self.encoder = MessageEncoder(message_length, wm_channels=wm_channels)
        self.decoder = MessageDecoder(message_length, wm_channels=wm_channels)
        
    def forward(self, x):
        raise NotImplementedError

    def embed(self, noise_latent, message):
        raise NotImplementedError

    def attack(self, freq_latent_wm, strength):
        raise NotImplementedError

    def extract(self, freq_latent_wm, freq_noise_attacked):
        raise NotImplementedError
    
    def save_model(self, model_path, model_info):
        epoch = model_info['epoch']
        latest_model = os.path.join(model_path, f"{WMModel.MODEL_BASE_NAME}_{epoch}.pth")
        model_info['model'] = self.state_dict()
        torch.save(model_info, latest_model)
        return latest_model
    
    def load_model(self, model_path, device=torch.device("cuda")):
        model_files = glob.glob(os.path.join(model_path, f"{WMModel.MODEL_BASE_NAME}_*.pth"))
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            model_dict = torch.load(latest_model, map_location=device)
            self.load_state_dict(model_dict['model'])
            model_dict.pop('model')
        return model_dict

class WMModel_DCT(WMModel):
    def __init__(self, message_length, wm_channels=1, wm_dim=32):
        super().__init__(message_length, wm_channels=wm_channels, wm_dim=wm_dim)

    def forward_transform(self, noise_latent):
        freq_latent = dct.dct_2d(noise_latent, norm='ortho')
        return freq_latent
        
    def embed(self, noise_latent, message):
        coeff_mod = self.encoder(message)
        freq_latent = dct.dct_2d(noise_latent, norm='ortho')
        freq_latent_wm = freq_latent.clone()
        freq_latent_wm[:, 3:, :self.wm_dim, :self.wm_dim] += coeff_mod
        return freq_latent_wm

    def attack(self, freq_latent_wm, strength):
        recon_noise = dct.idct_2d(freq_latent_wm, norm='ortho')
        noise_attacked = slerp(strength, recon_noise)
        freq_noise_attacked = dct.dct_2d(noise_attacked, norm='ortho')
        return freq_noise_attacked

    def extract(self, freq_latent_wm, freq_noise_attacked):
        recon_noise = dct.idct_2d(freq_latent_wm, norm='ortho')
        decoded_msg = self.decoder(freq_noise_attacked[:, 3:, :self.wm_dim, :self.wm_dim])
        return decoded_msg, recon_noise

class WMModel_FFT(WMModel):
    def __init__(self, message_length, wm_channels=1, wm_dim=32):
        super().__init__(message_length, wm_channels=wm_channels, wm_dim=wm_dim)

        self.encoder = MessageEncoder(message_length, wm_channels=wm_channels*2)
        self.decoder = MessageDecoder(message_length, wm_channels=wm_channels*2)

    def forward_transform(self, noise_latent):
        freq_latent = torch.fft.fftshift(torch.fft.fft2(noise_latent), dim=(-1, -2))
        return freq_latent

    def embed(self, noise_latent, message):
        coeff_mod = self.encoder(message)
        # 1. Split into real and imaginary parts
        coeff_mod_real = coeff_mod[:, 0:1, :, :] # shape [B, 1, wm_dim, wm_dim]
        coeff_mod_imag = coeff_mod[:, 1:2, :, :] # shape [B, 1, wm_dim, wm_dim]

        # 2. Create the complex perturbation
        mod_complex = torch.complex(coeff_mod_real, coeff_mod_imag)
        
        freq_latent = torch.fft.fftshift(torch.fft.fft2(noise_latent), dim=(-1, -2))
        
        B, C, M, N = freq_latent.shape
        cx, cy = N // 2, M // 2
        
        # 3. Define the "free half-region" strictly right of the vertical DC axis
        w_actual = min(self.wm_dim, N - (cx + 1)) 
        h_actual = min(self.wm_dim, M)
        y_start = cy - h_actual // 2
        y_end = y_start + h_actual
        x_start = cx + 1
        x_end = x_start + w_actual
        
        # 4. Create a zero delta mask and insert the complex watermark into channel 3
        delta = torch.zeros_like(freq_latent)
        mod_cropped = mod_complex[:, :, :h_actual, :w_actual]
        delta[:, 3:, y_start:y_end, x_start:x_end] = mod_cropped
        
        # 5. Enforce strict Hermitian Symmetry
        delta_unshift = torch.fft.ifftshift(delta, dim=(-1, -2))
        
        idx_M = torch.arange(M, device=delta.device)
        idx_N = torch.arange(N, device=delta.device)
        idx_M_rev = (-idx_M) % M
        idx_N_rev = (-idx_N) % N
        
        # Reflect and conjugate
        delta_reflected = delta_unshift[..., idx_M_rev[:, None], idx_N_rev]
        delta_sym_unshift = delta_unshift + torch.conj(delta_reflected)
        
        # Shift back and add to the latent
        delta_sym = torch.fft.fftshift(delta_sym_unshift, dim=(-1, -2))
        freq_latent_wm = freq_latent + delta_sym
        
        return freq_latent_wm

    def attack(self, freq_latent_wm, strength):
        recon_noise = torch.fft.ifft2(torch.fft.ifftshift(freq_latent_wm, dim=(-1, -2))).real
        noise_attacked = slerp(strength, recon_noise)
        freq_noise_attacked = torch.fft.fftshift(torch.fft.fft2(noise_attacked), dim=(-1, -2))
        return freq_noise_attacked
    
    def extract(self, freq_latent_wm, freq_noise_attacked):
        # 1. Reconstruct the spatial noise for generation
        recon_noise = torch.fft.ifft2(torch.fft.ifftshift(freq_latent_wm, dim=(-1, -2))).real
        
        B, C, M, N = freq_noise_attacked.shape
        cx, cy = N // 2, M // 2
        
        w_actual = min(self.wm_dim, N - (cx + 1))
        h_actual = min(self.wm_dim, M)
        y_start = cy - h_actual // 2
        y_end = y_start + h_actual
        x_start = cx + 1
        x_end = x_start + w_actual
        
        # 2. Extract the complex region from channel 3
        extracted_complex = freq_noise_attacked[:, 3:, y_start:y_end, x_start:x_end]
        
        # 3. Separate into real and imaginary, and stack into a 2-channel tensor
        extracted_mod = torch.cat([extracted_complex.real, extracted_complex.imag], dim=1)
        
        # 4. Pad if the free half-region truncated the rightmost column 
        if w_actual < self.wm_dim or h_actual < self.wm_dim:
            pad_w = self.wm_dim - w_actual
            pad_h = self.wm_dim - h_actual
            extracted_mod = F.pad(extracted_mod, (0, pad_w, 0, pad_h))
            
        decoded_msg = self.decoder(extracted_mod)
        
        return decoded_msg, recon_noise

def slerp(strength, org_latent):
    """
    對整批張量進行球面線性插值 (Batched Spherical Linear Interpolation)。
    內部自動產生目標高斯雜訊。
    
    參數:
        strength (Tensor): 插值強度，形狀為 [B, 1]。0 代表完全保留 org_latent，1 代表完全變成隨機雜訊。
        org_latent (Tensor): 原始潛在張量，形狀為 [B, 4, 64, 64]。
        
    返回:
        Tensor: 保持 N(0,1) 分佈的插值結果，形狀與 org_latent 相同 [B, 4, 64, 64]。
    """
    # 取得 Batch Size
    B = org_latent.shape[0]
    
    # 內部直接產生與 org_latent 同維度、同設備的純高斯雜訊
    low = org_latent
    high = torch.randn_like(org_latent)
    
    # 將空間維度攤平，保留 Batch 維度。形狀變為 [B, 16384]
    low_flat = low.reshape(B, -1)
    high_flat = high.reshape(B, -1)
    
    # 計算每個 batch 獨立的 L2 範數 (Norm)，保持維度以便廣播 [B, 1]
    low_norm = torch.norm(low_flat, dim=-1, keepdim=True)
    high_norm = torch.norm(high_flat, dim=-1, keepdim=True)
    
    # 將向量單位化
    low_unit = low_flat / low_norm
    high_unit = high_flat / high_norm
    
    # 計算每個 batch 的點積 (Cosine Similarity) [B, 1]
    # 對最後一個維度 (-1) 進行相乘後求和
    dot_product = torch.sum(low_unit * high_unit, dim=-1, keepdim=True)
    
    # 使用 clamp 確保數值穩定 (稍微縮限於 -0.9999 到 0.9999)
    # 避免浮點數誤差導致 acos 產生 NaN，且有利於如果需要反向傳播時的梯度穩定
    dot_product = torch.clamp(dot_product, -0.9999, 0.9999)
    
    # 計算夾角 (Omega) 與 sin(Omega) [B, 1]
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    
    # 處理極端情況：避免除以零
    # 建立一個遮罩 (Mask)，找出哪些 batch 的夾角極小，需要退化為一般線性插值 (Lerp)
    is_lerp = sin_omega < 1e-5
    
    # 把會出問題的 sin_omega 暫時替換為 1.0 (反正在下一步 is_lerp 為 True 的地方會被覆蓋掉)
    safe_sin_omega = torch.where(is_lerp, torch.ones_like(sin_omega), sin_omega)
    
    # 計算 Slerp 權重 [B, 1]
    weight_low_slerp = torch.sin((1.0 - strength) * omega) / safe_sin_omega
    weight_high_slerp = torch.sin(strength * omega) / safe_sin_omega
    
    # 計算 Lerp 權重 [B, 1]
    weight_low_lerp = 1.0 - strength
    weight_high_lerp = strength
    
    # 根據 is_lerp 遮罩，為每個 batch 挑選對應的權重 [B, 1]
    weight_low = torch.where(is_lerp, weight_low_lerp, weight_low_slerp)
    weight_high = torch.where(is_lerp, weight_high_lerp, weight_high_slerp)
    
    # 將權重形狀擴展為 [B, 1, 1, 1] 以匹配原張量，以便進行廣播 (Broadcasting)
    weight_low = weight_low.reshape(B, 1, 1, 1)
    weight_high = weight_high.reshape(B, 1, 1, 1)
    
    # 套用權重並返回結果，形狀自動推導回 [B, 4, 64, 64]
    return weight_low * low + weight_high * high

def exec_train(args):
    import logging

    model = args.WM_MODEL[args.type](args.msg_len, wm_dim=args.wm_dim).to(args.device)

    # Set up optimizer and loss functions
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    model.train()

    # Setup for checkpoint management
    prev_model_path = None

    start_epoch = 0
    best_epoch = -1
    best_loss = float('inf')
    if args.resume is not None:
        model_info = model.load_model(args.resume)
        start_epoch = model_info['epoch']
        best_epoch = start_epoch
        best_loss = model_info['epoch_loss']
        logging.info(f"Resuming training from {args.resume} at epoch {start_epoch}, loss {best_loss:.6f}")
    else:
        model_info = {'epoch': -1, 'epoch_loss': float('inf')}

    logging.info(f"Starting training on {args.device}...")

    epoch_time = AverageMeter()
    for epoch in range(start_epoch, args.epochs):
        train_log = defaultdict(AverageMeter)
        pbar_train = tqdm(range(args.steps), ncols=100, desc=f'Epoch {epoch+1}')
        update_interval = max(len(pbar_train) // 10, 100)
        for step in pbar_train:
            # 1. Generate random batch of Gaussian latents
            noise_latent = torch.randn(args.batch_size, 4, 64, 64, device=args.device)
            message = torch.Tensor(np.random.choice([0, 1], (args.batch_size, args.msg_len))).to(args.device)

            freq_latent_wm = model.embed(noise_latent, message)

            if args.strength > 0.0:
                strength = torch.rand(args.batch_size, 1).to(args.device) * args.strength
                freq_noise_attacked = model.attack(freq_latent_wm, strength)
            else:
                freq_noise_attacked = freq_latent_wm
            
            decoded_msg, recon_noise = model.extract(freq_latent_wm, freq_noise_attacked)
            
            # 6. Calculate losses
            wm_loss = bce_loss(decoded_msg, message)
            recon_loss = mse_loss(noise_latent, recon_noise)
            total_loss = 10.0 * wm_loss + recon_loss
            
            train_log['total_loss'].update(total_loss.detach().item())
            train_log['wm_loss'].update(wm_loss.detach().item())
            train_log['recon_loss'].update(recon_loss.detach().item())
            
            bit_error_rate = (decoded_msg.sigmoid().round() != message).float().mean()
            train_log['bit_error_rate'].update(bit_error_rate.detach().item())

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # logging.info progress
            if False and step % update_interval == update_interval//2:
                metric_str = ', '.join([f'{m}: {train_log[m].avg:.6f}' for m in train_log])
                logging.info(f"\n\t[{epoch+1}/{args.epochs}] {metric_str}")

        epoch_time.update(pbar_train.format_dict['elapsed'])
        est_remain = epoch_time.avg * (args.epochs - epoch)
        str_complete = (datetime.datetime.now() + datetime.timedelta(seconds=est_remain)).strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f'Estimate to complete at: {Fore.YELLOW}{Style.BRIGHT}{str_complete}{Style.RESET_ALL}')        
        
        avg_loss = train_log['total_loss'].avg
        if avg_loss < best_loss:
            # Save current epoch models
            model_info['epoch'] = epoch+1
            model_info['epoch_loss'] = avg_loss
            latest_model = model.save_model(args.this_run_folder, model_info)
            logging.info(f"Epoch {epoch+1} loss {avg_loss:.6f} < {best_loss:.6f}, saving model {latest_model}")
            best_loss = avg_loss
            best_epoch = epoch+1

            # Delete previous epoch models to save space
            if prev_model_path and os.path.exists(prev_model_path):
                os.remove(prev_model_path)

            prev_model_path = latest_model
        else:
            logging.info(f"Epoch {epoch+1} loss {avg_loss:.6f} did not improve from {best_loss:.6f}")

        metric_str = ', '.join([f'{m}: {train_log[m].avg:.6f}' for m in train_log])
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] [{best_epoch}] {metric_str}")

def inverse_image_batch(image, pipe):
    batch_size = image.shape[0]
    image_tensor = 2.0 * image - 1.0
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.mode()
        # Important: Scale latents by the magic number constant
        latents = latents * pipe.vae.config.scaling_factor
    
        prompt = ""
        text_input = pipe.tokenizer([prompt] * batch_size, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        encoder_hidden_states = pipe.text_encoder(text_input.input_ids.to(args.device))[0]

        # Setup the Inverse Scheduler
        num_inference_steps = 50
        inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config, clip_sample=False)
        inverse_scheduler.set_timesteps(num_inference_steps, device=args.device)

        print("Starting DDIM Inversion...")

        # C. The Inversion Loop (t: 0 -> 1000)
        inverted_latents = latents.clone()
        for t in tqdm(inverse_scheduler.timesteps):
            # 1. Predict noise
            noise_pred = pipe.unet(inverted_latents, t, encoder_hidden_states=encoder_hidden_states).sample
            # 2. Step "backwards" (adding noise deterministically)
            inverted_latents = inverse_scheduler.step(noise_pred, t, inverted_latents).prev_sample
    return inverted_latents

from main.wmattacker import *
from main.attdiffusion import ReSDPipeline

def exec_embed(args):
    att_pipe = ReSDPipeline.from_pretrained(LDM_MODELS['sd15'], torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    att_pipe.set_progress_bar_config(disable=True)
    att_pipe.to(args.device)
    
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_path = args.resume.split('/')[-1]
    excel_path = os.path.join('.', 'metrics.xlsx')

    attackers = {
        'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=args.device),
        'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=args.device),
        'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
        'jpeg_attacker_50': JPEGAttacker(quality=50),
        #'rotate_90': RotateAttacker(degree=90),
        'brightness_0.5': BrightnessAttacker(brightness=0.5),
        'contrast_0.5': ContrastAttacker(contrast=0.5),
        'Gaussian_noise': GaussianNoiseAttacker(std=0.1),
        'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=2),
        #'bm3d': BM3DAttacker(),
    }
    model = args.WM_MODEL[args.type](args.msg_len, wm_dim=args.wm_dim).to(args.device)

    if args.resume is not None:
        model_info = model.load_model(args.resume)
        print(f"Load model from {args.resume} at epoch {model_info['epoch']}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    model_id = LDM_MODELS[args.ldm]
    # Load Stable Diffusion pipeline
    print(f"Loading Stable Diffusion {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.to(args.device)
    
    metrics = defaultdict(AverageMeter)
    perf = defaultdict(AverageMeter)
    records = []
    iter_time = AverageMeter()
    for exp_id in range(args.expr):
        start_time = time.time()
        rand_seed = exp_id + 20250508
        fn_base = f'trans_embed_{exp_id:04d}.png'

        # Select prompt and sample noise
        prompt = args.prompts[exp_id]
        print(f"Selected prompt: {prompt}")
        noise_latent = torch.randn(1, 4, 64, 64, device=args.device, generator=torch.Generator(device=args.device).manual_seed(rand_seed))

        rng = np.random.default_rng(rand_seed)
        message = torch.Tensor(rng.choice([0, 1], (1, args.msg_len))).to(args.device)
        # Embed and reconstruct
        freq_latent_wm = model.embed(noise_latent, message)
        _, recon_noise = model.extract(freq_latent_wm, freq_latent_wm)

        embed_mse = F.mse_loss(recon_noise, noise_latent).item()
        perf['embed_mse'].update(embed_mse)

        var = recon_noise.var(unbiased=False)
        mu = recon_noise.mean()
        perf['embed_mean'].update(mu.item())
        perf['embed_var'].update(var.item())
        # Formula: D_KL( N(\mu, \sigma^2) || N(0, 1) )
        kl_div = 0.5 * (var + mu**2 - 1.0 - torch.log(var + 1e-8))
        perf['embed_kl_div'].update(kl_div.item())

        # Batch the latents to generate images in one forward pass
        batched_latents = torch.cat([noise_latent, recon_noise], dim=0)
        with torch.no_grad():
            # Generate both original and watermarked images
            images = pipe([prompt] * 2, latents=batched_latents).images
            image_org, image_wm = images[0], images[1]

        org_name = make_filepath(fn_base, dir_name=args.run_folder_image, tag='org')
        image_org.save(org_name)

        wm_name = make_filepath(fn_base, dir_name=args.run_folder_image, tag='wm')
        image_wm.save(wm_name)

        psnr = calculate_psnr(image_org, image_wm)

        out_name = make_filepath(fn_base, dir_name=args.run_folder_image, tag='comp')
        to_tensor = torchvision.transforms.ToTensor()
        print(f"Saved {out_name}, PSNR: {psnr:.2f} dB")
        torchvision.utils.save_image(torch.stack([to_tensor(image_org), to_tensor(image_wm)]), out_name)
        
        inverse_names = [wm_name]
        md_tags = ['wm']
        
        wm_name = make_filepath(fn_base, dir_name=args.run_folder_image, tag='wm')
        for atk_name in attackers:
            edit_name = make_filepath(fn_base, dir_name=args.run_folder_image, tag=atk_name)
            attackers[atk_name].attack([wm_name], [edit_name], multi=True)
            inverse_names.append(edit_name)
            md_tags.append(atk_name)
        
        # Read the images from inverse_names and make it as a image tensor of shape [B, 3, 512, 512]
        img_list = []
        for im_path in inverse_names:
            img = Image.open(im_path).convert("RGB")
            img_list.append(to_tensor(img))
        img_batch = torch.stack(img_list).to(args.device)
        inverted_latents = inverse_image_batch(img_batch, pipe)

        for i, im_name in enumerate(inverse_names):
            latents = inverted_latents[i:i+1:, :, :, :]
            with torch.no_grad():
                freq_latent = model.forward_transform(latents)
                decoded_msg, _ = model.extract(freq_latent, freq_latent)
                bit_error_rate = (decoded_msg.sigmoid().round() != message).float().mean().item()
                print(f"Extract from {im_name.split('/')[-1]}, bit error rate: {bit_error_rate:.6f}")
                metrics[md_tags[i]].update(bit_error_rate)
                records.append({
                    'experiment_time': experiment_id,
                    'model': f'{args.ldm}:{model_path}',
                    'filename': im_name.split('/')[-1], 
                    'attack': md_tags[i], 'bit_error_rate': bit_error_rate
                })

        iter_time.update(time.time() - start_time)
        est_remain = iter_time.avg * (args.expr - 1 - exp_id)
        str_complete = (datetime.datetime.now() + datetime.timedelta(seconds=est_remain)).strftime("%Y-%m-%d %H:%M:%S")
        print(f'Estimate to complete at: {Fore.YELLOW}{Style.BRIGHT}{str_complete}{Style.RESET_ALL}')

    new_df = pd.DataFrame(records)
    new_df.set_index('experiment_time', inplace=True)

    summary_record = {
        'experiment_time': experiment_id,
        'model': f'{args.ldm}:{model_path}',
        'embed_mse': perf['embed_mse'].avg,
        'embed_mean': perf['embed_mean'].avg,
        'embed_var': perf['embed_var'].avg,
        'embed_kl_div': perf['embed_kl_div'].avg,
    }
    for i in metrics:
        print(f"{i}: {metrics[i].avg:.6f}")
        perf['average'].update(metrics[i].avg)
        summary_record[i] = metrics[i].avg
    print(f"Average: {perf['average'].avg:.6f}")
    summary_record['Average'] = perf['average'].avg

    summary_df = pd.DataFrame([summary_record])
    summary_df.set_index('experiment_time', inplace=True)

    if os.path.exists(excel_path):
        try:
            old_df = pd.read_excel(excel_path, sheet_name='Records', index_col=0)
            final_df = pd.concat([old_df, new_df])
        except Exception:
            try:
                old_df = pd.read_excel(excel_path, index_col=0)
                final_df = pd.concat([old_df, new_df])
            except Exception as e:
                print(f"讀取舊檔失敗: {e}，將建立新檔。")
                final_df = new_df
                
        try:
            old_summary_df = pd.read_excel(excel_path, sheet_name='Summary', index_col=0)
            final_summary_df = pd.concat([old_summary_df, summary_df])
        except Exception:
            final_summary_df = summary_df
            
        print(f"找到舊檔案，已追加數據 (Records 目前總計 {len(final_df)} 筆)")
    else:
        print("尚未發現舊檔，正在建立新的 metrics.xlsx...")
        final_df = new_df
        final_summary_df = summary_df

    # 4. 儲存檔案
    with pd.ExcelWriter(excel_path) as writer:
        final_df.to_excel(writer, sheet_name='Records', index=True)
        final_summary_df.to_excel(writer, sheet_name='Summary', index=True)
    print(f"數據已更新至: {excel_path}")

def exec_variety(args):
    model = args.WM_MODEL[args.type](args.msg_len, wm_dim=args.wm_dim).to(args.device)

    if args.resume is not None:
        model_info = model.load_model(args.resume)
        print(f"Load model from {args.resume} at epoch {model_info['epoch']}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    # Load Stable Diffusion pipeline
    print(f"Loading Stable Diffusion {model_id}...")
    model_id = LDM_MODELS[args.ldm]
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.to(args.device)

    var_folder = os.path.join(args.this_run_folder, 'variety')
    ensure_dir(var_folder)

    mse_loss = nn.MSELoss()

    NUM_SAMP = 10
    selected = ['0086', '0008', '0011', '0010', '0006', '0081', '0066', '0007', '0056', '0087', '0092']
    for s in selected:
        exp_id = int(s)
        prompt = args.prompts[exp_id]
        print(f"[{exp_id:04d}] Selected prompt: {prompt}")

        rand_seed = exp_id + 20250508
        noise_latent = torch.randn(1, 4, 64, 64, device=args.device, generator=torch.Generator(device=args.device).manual_seed(rand_seed))
        noise_latent = noise_latent.repeat(NUM_SAMP, 1, 1, 1)

        rng = np.random.default_rng(rand_seed)
        message = torch.Tensor(rng.choice([0, 1], (NUM_SAMP, args.msg_len))).to(args.device)

        freq_latent_wm = model.embed(noise_latent, message)
        decoded_msg, recon_noise = model.extract(freq_latent_wm, freq_latent_wm)

        with torch.no_grad():
            images = pipe([prompt] * NUM_SAMP, latents=recon_noise).images

        fn_base = f'variety_{exp_id:04d}.png'
        for i in range(NUM_SAMP):
            var_name = make_filepath(fn_base, dir_name=var_folder, ext_name='png', tag=f'{i:02d}')
            images[i].save(var_name)

            r, n = recon_noise[i:i+1, :, :, :], noise_latent[i:i+1, :, :, :]
            m = r.mean().item()
            v = r.var().item()
            mse = mse_loss(n, r).item()
            bit_error_rate = (decoded_msg[i:i+1].sigmoid().round() != message[i:i+1]).float().mean().item()
            print(f"{var_name}: mean {m:.6f}, var {v:.6f}, mse {mse:.6f}, bit_error_rate {bit_error_rate:.6f}")

def init_prepare(args):
    args.device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    
    args.prompts = load_prompts_from_file(args.prompts)
    print(f"Loaded {len(args.prompts)} prompts")
    
    args.WM_MODEL = {
        'DCT': WMModel_DCT,
        'FFT': WMModel_FFT
    }
    print(f'Using model: {args.WM_MODEL[args.type].__name__}')

    if args.resume is None:
        args.name = args.name.replace('_', '-')
        run_label = f'{args.name}_{args.type},m{args.msg_len},s{args.strength:.1f}_{time.strftime("%m%d-%H%M%S")}'
    else:
        run_label = args.resume.split('/')[-1]
    args.this_run_folder = os.path.join(args.run_folder, run_label)
    args.run_folder_image = os.path.join(args.this_run_folder, f'images/{args.ldm}')
    ensure_dir(args.this_run_folder)
    ensure_dir(args.run_folder_image)
    
    if args.dispatch == exec_train:
        import logging
        log_name = os.path.join(args.this_run_folder, f'{run_label}.log')
        logging.basicConfig(level=logging.INFO, format='%(message)s',
                            handlers=[
                                logging.FileHandler(log_name),
                                logging.StreamHandler(sys.stdout)
                            ])
        logging.info(f'Start with run folder: {args.this_run_folder}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--train', dest='dispatch', action='store_const', const=exec_train, default=None, help='train')
    parser.add_argument('--embed', dest='dispatch', action='store_const', const=exec_embed, help='embed watermark and generate image')
    parser.add_argument('--var', dest='dispatch', action='store_const', const=exec_variety, help='')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument('--name', type=str, default='default', required=False, help='name the training')
    parser.add_argument('--msg_len', '-m', default=WM_MESSAGE_LENGTH, type=int, help='The length in bits of the watermark.')
    parser.add_argument('--wm_dim', default=32, type=int, help='')
    parser.add_argument('--strength', type=float, default=0.6, help='')
    parser.add_argument('--type', type=str, choices=['DCT', 'FFT'], default='DCT', help='')
    parser.add_argument('--ldm', type=str, choices=list(LDM_MODELS.keys()), default='sd15', help='')
    parser.add_argument('--expr', default=100, type=int, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--steps', type=int, default=5000, help='number of steps')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--output', '-o', default='output', type=str, help='output folder')
    parser.add_argument('--run_folder', default='logger', type=str, help='The output run folder')
    parser.add_argument('--resume', default=None, type=str, help='resume training from latest checkpoint')
    parser.add_argument('--prompts', required=False, default='prompts.txt', type=str, help='')

    args = parser.parse_args()

    if args.dispatch is None:
        parser.print_help()
    else:
        init_prepare(args)
        args.dispatch(args)
