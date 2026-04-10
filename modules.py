import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperpixelGAT(nn.Module):
    def __init__(self, in_channels, n_spix=100, hidden_dim=256):
        super().__init__()
        self.n_spix = n_spix
        
        self.ssn_proj = nn.Conv2d(20, 64, 1) 
        
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, 1)
        
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, 1)
        
        self.num_heads = 4
        self.head_dim = hidden_dim // 4
        self.scale = self.head_dim ** -0.5
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ssn_feat):
        """
        x: Backbone Feature (B, C, H, W)
        ssn_feat: SSN Output Feature (B, 25, H, W)
        """
        B, C, H, W = x.shape
        residual = x 
        
        if ssn_feat.shape[-2:] != (H, W):
            ssn_feat = F.interpolate(ssn_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        ssn_feat = self.ssn_proj(ssn_feat) # (B, 64, H, W)
        ssn_flat = ssn_feat.flatten(2).transpose(1, 2) # (B, HW, 64)
        
        num_w = int(self.n_spix**0.5)
        sp_centers = F.interpolate(ssn_feat, size=(num_w, num_w), mode='bilinear', align_corners=False)
        sp_centers_flat = sp_centers.flatten(2).transpose(1, 2) # (B, N_spix, 64)
        
        affinity_logits = torch.bmm(ssn_flat, sp_centers_flat.transpose(1, 2))
        
        affinity_logits = affinity_logits * (64 ** -0.5)
        
        affinity = F.softmax(affinity_logits, dim=2) # Q
        
        x_in = self.conv_in(x).flatten(2).transpose(1, 2) # (B, HW, Hid)
        
        Q_t = affinity.transpose(1, 2)
        nodes = torch.bmm(Q_t, x_in)

        q = self.q_linear(nodes).view(B, self.n_spix, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(nodes).view(B, self.n_spix, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(nodes).view(B, self.n_spix, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        nodes_enhanced = (attn @ v).transpose(1, 2).reshape(B, self.n_spix, -1)
        
        x_context = torch.bmm(affinity, nodes_enhanced)
        
        x_context = x_context.transpose(1, 2).reshape(B, -1, H, W)
        x_context = self.conv_out(x_context) # (B, C, H, W)
        
        combined = torch.cat([residual, x_context], dim=1) # (B, 2C, H, W)
        out = self.fusion_conv(combined) # (B, C, H, W)
        
        return out