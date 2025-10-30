import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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
        
        # --- SSTB Block ---
        self.sstb = SSTB(
            dim=num_feature,
            num_heads=4,
            window_size=8,
            mlp_ratio=4.0,
            dropout=0.1
        )
        
        # --- 1×1 Conv Fusion (like HMFormer) ---
        self.conv_1x1_1 = nn.Conv1d(2*num_feature, num_feature, 1)
        self.conv_1x1_2 = nn.Conv1d(2*num_feature, num_feature, 1)
        self.conv_1x1_3 = nn.Conv1d(2*num_feature, num_feature, 1)
        
        # --- Transformer-based encoding/decoding ---
        self.T_E = Transformer_E(num_feature)
        self.T_D = Transformer_D(num_feature)
        
        # --- Embedding ---
        self.Embedding = nn.Sequential(
            nn.Linear(num_channel + 3, num_feature),
        )

        # --- Refine (spectral-spatial reconstruction) ---
        self.refine = nn.Sequential(
            nn.Conv2d(num_feature, num_feature, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            MambaBlock(num_feature),
            nn.Conv2d(num_feature, num_channel, 3, 1, 1)
        )

    def forward(self, HSI, MSI):
        UP_LRHSI = F.interpolate(HSI, scale_factor=4, mode='bicubic', align_corners=False)
        UP_LRHSI = UP_LRHSI.clamp_(0, 1)
        H, W = UP_LRHSI.size(2), UP_LRHSI.size(3)

        fused = torch.cat((UP_LRHSI, MSI), dim=1)                     # [B, S+s, H, W]
        seq = rearrange(fused, 'B C H W -> B (H W) C', H=H, W=W)      # [B, HW, C]
        X0 = self.Embedding(seq)                                      # [B, HW, num_feature]
        X1 = self.sstb(X0, H, W)
        X2 = self.sstb(X1, H, W)
        X3 = self.sstb(X2, H, W)
        X4 = self.sstb(X3, H, W)
        s13 = torch.cat((X4, X3), dim=-1)                            # [B, HW, 2*num_feature]
        s13_t = s13.transpose(1, 2)                                  # [B, 2*num_feature, HW]
        s13_reduced = self.conv_1x1_1(s13_t)                         # [B, num_feature, HW]
        S1 = s13_reduced.transpose(1, 2)                             # [B, HW, num_feature]
        s12 = torch.cat((S1, X2), dim=-1)                            # [B, HW, 2*num_feature]
        s12_t = s12.transpose(1, 2)                                  # [B, 2*num_feature, HW]
        s12_reduced = self.conv_1x1_2(s12_t)                         # [B, num_feature, HW]
        S2 = s12_reduced.transpose(1, 2)                             # [B, HW, num_feature]
        s11 = torch.cat((S2, X1), dim=-1)                            # [B, HW, 2*num_feature]
        s11_t = s11.transpose(1, 2)                                  # [B, 2*num_feature, HW]
        s11_reduced = self.conv_1x1_3(s11_t)                         # [B, num_feature, HW]
        S3 = s11_reduced.transpose(1, 2)                             # [B, HW, num_feature] 
        Highpass = rearrange(S3, 'B (H W) C -> B C H W', H=H, W=W)  # [B, num_feature, H, W]

        Highpass = self.refine(Highpass)                             # [B, num_channel, H, W]

        output = Highpass + UP_LRHSI                                 # refined HR-HSI
        output = output.clamp_(0, 1)                                 # restrict pixel range [0, 1]

        return output, UP_LRHSI, Highpass


# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer_E(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48, sp_sz=64*64, num_channels = 48,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer_D(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48 , sp_sz=64*64, num_channels = 48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        for attn1,attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, mask=mask)
            x = ff(x)
        return x