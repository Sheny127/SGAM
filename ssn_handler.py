import torch
import torch.nn as nn
import torch.nn.functional as F
from ssn_model import SSNModel

def rgb_to_lab(image):
    mask = (image > 0.04045).float()
    image = ( (image + 0.055) / 1.055 ) ** 2.4 * mask + image / 12.92 * (1 - mask)
    
    xyz = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], device=image.device)
    
    b, c, h, w = image.shape
    image_reshaped = image.view(b, c, -1)
    xyz_img = torch.matmul(xyz, image_reshaped).view(b, 3, h, w)
    
    xyz_ref = torch.tensor([0.950456, 1.0, 1.088754], device=image.device).view(1, 3, 1, 1)
    xyz_img = xyz_img / xyz_ref
    
    mask = (xyz_img > 0.008856).float()
    xyz_f = xyz_img ** (1/3) * mask + (7.787 * xyz_img + 16/116) * (1 - mask)
    
    L = (116 * xyz_f[:, 1:2, :, :] - 16)
    a = 500 * (xyz_f[:, 0:1, :, :] - xyz_f[:, 1:2, :, :])
    b = 200 * (xyz_f[:, 1:2, :, :] - xyz_f[:, 2:3, :, :])
    
    return torch.cat([L, a, b], dim=1)

class FrozenSSN(nn.Module):
    def __init__(self, weight_path, nspix=100, n_iter=10, fdim=20):
        super().__init__()
        self.model = SSNModel(feature_dim=fdim, nspix=nspix, n_iter=n_iter)
        
        print(f"loading SSN weight: {weight_path}")
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"warning weight load fail ({e})")
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.nspix = nspix

    def forward(self, x):
        # x: (B, 3, H, W) RGB [0, 1]
        
        x_lab = rgb_to_lab(x)

        b, c, h, w = x.shape
        coords = torch.stack(torch.meshgrid(
            torch.arange(h, device=x.device), 
            torch.arange(w, device=x.device), indexing='ij'), 0)
        coords = coords[None].float().repeat(b, 1, 1, 1)
        
        inputs = torch.cat([x_lab, coords], 1) 

        with torch.no_grad():
            pixel_f = self.model.feature_extract(inputs)
            
        return pixel_f