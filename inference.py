import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Generator_EL_GAN(nn.Module):
    def __init__(self, n_iter=3):
        super(Generator_EL_GAN, self).__init__()
        self.n_iter = n_iter
        self.conv1 = DepthwiseSeparableConv(3, 32)
        self.conv2 = DepthwiseSeparableConv(32, 32)
        self.conv3 = DepthwiseSeparableConv(32, 32)
        self.conv4 = DepthwiseSeparableConv(32, 32)
        self.conv5 = DepthwiseSeparableConv(64, 32)
        self.conv6 = DepthwiseSeparableConv(64, 32)
        self.conv_final = nn.Conv2d(64, 3 * n_iter, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        A = 2.0 * torch.tanh(self.conv_final(torch.cat([x1, x6], 1)))
        
        enhanced = x
        for i in range(self.n_iter):
            alpha = A[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + alpha * (torch.pow(enhanced, 2) - enhanced)
        return enhanced


def run_enhancement(image_path, model_path, output_path=None, show=True):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    model = Generator_EL_GAN(n_iter=3).to(device)
    if not os.path.exists(model_path):
        print(f"[!] Error: Model weights not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    print(f"[*] Enhancing image: {os.path.basename(image_path)}")
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)
        enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)

    enhanced_img = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu())
    enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)

    if output_path:
        enhanced_img.save(output_path)
        print(f"[*] Saved enhanced image to: {output_path}")

    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Low-Light Input")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_img)
        plt.title("EL-GAN Enhanced")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EL-GAN Low-Light Image Enhancement Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to the low-light input image")
    parser.add_argument("--weights", type=str, default="generator_finetuned.pth", help="Path to the trained .pth weights")
    parser.add_argument("--output", type=str, default="enhanced_output.jpg", help="Path to save the enhanced image")
    parser.add_argument("--no_show", action="store_true", help="Do not display the result window")

    args = parser.parse_args()

    run_enhancement(
        image_path=args.input, 
        model_path=args.weights, 
        output_path=args.output, 
        show=not args.no_show
    )
