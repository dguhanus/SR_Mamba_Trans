import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from bidirectional_cross_attention import BidirectionalCrossAttention
import random
# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# ---------------------SSTB Implementation-------------------------
from mamba_ssm import Mamba  

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,     # input/output feature dim
            d_state=d_state, # state space size
            d_conv=d_conv,   # local conv size
            expand=expand    # expansion ratio
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)   
        x = self.mamba(x)                  
        x = x.transpose(1, 2).view(b, c, h, w)
        return x
class SSTB(nn.Module):
    """
    Spatial-Spectral Transformer Block (SSTB)
    Contains Spectral-wise MSA, Spatial-wise MSA, and FFN
    """
    def __init__(self, dim, num_heads=8, window_size=8, mlp_ratio=4.0, dropout=0.1):
        super(SSTB, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Spectral-wise Multi-head Self-Attention
        self.spe_msa = SpectralMSA(dim, num_heads, dropout)
        
        # Spatial-wise Multi-head Self-Attention
        self.spa_msa = SpatialMSA(dim, num_heads, window_size, dropout)
        
        # Feed Forward Network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor (B, H*W, C)
            H: Height of feature map
            W: Width of feature map
        Returns:
            Output tensor (B, H*W, C)
        """
        # Spectral-wise MSA with residual connection
        x = x + self.spe_msa(self.norm1(x), H, W)
        
        # Spatial-wise MSA with residual connection
        x = x + self.spa_msa(self.norm2(x), H, W)
        
        # FFN with residual connection
        x = x + self.ffn(self.norm3(x))
        
        return x

class SpectralMSA(nn.Module):
    """Spectral-wise Multi-head Self-Attention"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable reweighting parameters
        self.alpha = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Transpose for spectral attention
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        heads = []
        for i in range(self.num_heads):
            # Attention along spectral dimension
            attn = torch.matmul(k[:, i], q[:, i].transpose(-2, -1))
            attn = attn / self.alpha[i]
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            head_out = torch.matmul(attn, v[:, i])
            heads.append(head_out.transpose(-2, -1))
        
        out = torch.cat(heads, dim=-1)
        out = self.proj(out)
        
        return out

class SpatialMSA(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def window_partition(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = H + pad_h, W + pad_w
        
        x = x.view(B, H_pad // self.window_size, self.window_size,
                W_pad // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, self.window_size * self.window_size, C)
        
        return windows, H_pad, W_pad
    
    def window_reverse(self, windows, H_pad, W_pad, H, W):
        """Reverse window partition"""
        B_win = windows.shape[0]
        num_windows_h = H_pad // self.window_size
        num_windows_w = W_pad // self.window_size
        B = B_win // (num_windows_h * num_windows_w)
        C = windows.shape[-1]
        
        x = windows.view(B, num_windows_h, num_windows_w,
                        self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, C)
        
        if H_pad > H or W_pad > W:
            x = x[:, :H, :W, :].contiguous()
        
        return x.view(B, H * W, C)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Partition into windows
        x_windows, H_pad, W_pad = self.window_partition(x, H, W)
        B_win, win_size, C = x_windows.shape
        
        # Generate Q, K, V for each window
        qkv = self.qkv(x_windows).reshape(B_win, win_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply attention within each window
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x_attn = torch.matmul(attn, v).transpose(1, 2).reshape(B_win, win_size, C)
        x_attn = self.proj(x_attn)
        
        # Reverse window partition
        x = self.window_reverse(x_attn, H_pad, W_pad, H, W)
        
        return x

# --------------------------Main------------------------------- #

# --------------------------Main------------------------------- #
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        num_channel = 31
        num_feature = 48
        
        # Keep SSTB
        self.sstb = SSTB(
            dim=num_feature,
            num_heads=4,
            window_size=8,
            mlp_ratio=4.0,
            dropout=0.1
        )

        # Replace BCAT with Mamba
        self.mamba = MambaBlock(num_feature)

        # Learnable weights
        self.x1 = nn.Parameter(torch.randn(1))
        self.x2 = nn.Parameter(torch.randn(1))
        
 
        self.Embedding = nn.Sequential(
            nn.Linear(num_channel+3, num_feature),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(num_feature, num_feature, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            MambaBlock(num_feature),  
            nn.Conv2d(num_feature, num_channel, 3, 1, 1)
        )

    def forward(self, HSI, MSI):
        UP_LRHSI = F.interpolate(HSI, scale_factor=4, mode='bicubic')
        UP_LRHSI = UP_LRHSI.clamp_(0, 1)
        sz = UP_LRHSI.size(2)

        Data = torch.cat((UP_LRHSI, MSI), 1)
        E = rearrange(Data, 'B c H W -> B (H W) c', H=sz)
        E = self.Embedding(E)

        # --- SSTB branch ---
        E1 = self.sstb(E, sz, sz)

        # --- Mamba branch ---
        E2 = self.mamba(rearrange(E, 'B (H W) C -> B C H W', H=sz, W=sz))
        E2 = rearrange(E2, 'B C H W -> B (H W) C')

        # --- Combine ---
        E = self.x1 * E1 + self.x2 * E2
        Highpass = E

        Highpass = rearrange(Highpass, 'B (H W) C -> B C H W', H=sz)
        Highpass = self.refine(Highpass)
        output = Highpass + UP_LRHSI
        output = output.clamp_(0, 1)

        return output, UP_LRHSI, Highpass
