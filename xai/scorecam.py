import torch
import torch.nn.functional as F
import numpy as np


class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor):
        self.model.eval()

        with torch.no_grad():
            _ = self.model(input_tensor)

        activations = self.activations  # [1, C, H, W]
        b, k, u, v = activations.size()

        scores = []

        for i in range(k):
            activation_map = activations[:, i:i+1, :, :]
            upsampled = F.interpolate(
                activation_map,
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            norm = (upsampled - upsampled.min()) / (upsampled.max() + 1e-8)

            masked_input = input_tensor * norm

            output = self.model(masked_input)
            score = output.max().item()

            scores.append(score)

        weights = torch.tensor(scores).to(input_tensor.device)

        cam = torch.zeros((u, v)).to(input_tensor.device)

        for i in range(k):
            cam += weights[i] * activations[0, i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()