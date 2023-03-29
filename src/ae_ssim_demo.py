import os
import cv2

import numpy as np
from PIL import Image
import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import TestDataset
from model.auto_encorder import AutoEncoder
from image_processor import merge_patch, make_residual_map, make_heatmap, make_ssim_map

logs = "../logs/demo_auto_encoder_mse_ssim"
anomaly_images_path = "../../data/dataset/anormal/"
#model_param = "../models/AutoEncoder_SSIM_w160_d128.pth"
model_param = "../logs/AE_Checkpoint_MSE/conv_autoencoder_50000.pth"

os.makedirs(logs, exist_ok=True)

model = AutoEncoder(d=128).cuda()
model.load_state_dict(torch.load(model_param, map_location="cuda"))

for n, image_file in enumerate(os.listdir(anomaly_images_path)):
    print(f"processing {image_file}")

    trg_image = Image.open(os.path.join(anomaly_images_path,image_file))
    dataset = TestDataset(trg_image)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    input, _  = iter(dataloader).next()

    with torch.no_grad():
        output = model(input.cuda())
    
    output = np.clip((output*255).detach().cpu().numpy(),0,255).astype('uint8')
    patch_images = [output[i].transpose(1,2,0)[:,:,::-1] for i in range(output.shape[0])]
    
    decode_image = merge_patch(patch_images, 160, 128, 2, 2)
    cv_image = np.array(trg_image, dtype=np.uint8)[:,:,::-1]

    #residual_map = make_residual_map(cv_image, decode_image)
    residual_map = make_ssim_map(cv_image, decode_image)

    heat_map = make_heatmap(residual_map)

    cv2.imwrite(os.path.join(logs, f"{n}_input.jpg"), cv_image)
    cv2.imwrite(os.path.join(logs, f"{n}_decode.jpg"), decode_image)
    cv2.imwrite(os.path.join(logs, f"{n}_residual_map.jpg"), residual_map)
    cv2.imwrite(os.path.join(logs, f"{n}_heat_map.jpg"), heat_map)


