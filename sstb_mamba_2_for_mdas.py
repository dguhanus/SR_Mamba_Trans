import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


# -----------------------------------------------------------
# Spectral Attention
# -----------------------------------------------------------

class SpectralMSA(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.alpha = nn.Parameter(torch.ones(num_heads))

        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H, W):

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)

        q,k,v = qkv

        q = q.transpose(-2,-1)
        k = k.transpose(-2,-1)
        v = v.transpose(-2,-1)

        outs = []

        for i in range(self.num_heads):

            attn = torch.matmul(k[:,i], q[:,i].transpose(-2,-1))
            attn = attn / (self.alpha[i] + 1e-6)

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            o = torch.matmul(attn, v[:,i]).transpose(-2,-1)

            outs.append(o)

        x = torch.cat(outs, dim=-1)

        x = self.proj(x)

        return x


# -----------------------------------------------------------
# Spatial Window Attention
# -----------------------------------------------------------

class SpatialMSA(nn.Module):

    def __init__(self, dim, num_heads=4, window_size=8):

        super().__init__()

        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):

        B, N, C = x.shape

        x = x.view(B,H,W,C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h or pad_w:
            x = F.pad(x,(0,0,0,pad_w,0,pad_h))

        Hp,Wp = x.shape[1],x.shape[2]

        x = x.view(
            B,
            Hp//self.window_size,
            self.window_size,
            Wp//self.window_size,
            self.window_size,
            C
        )

        x = x.permute(0,1,3,2,4,5).contiguous()
        x = x.view(-1,self.window_size*self.window_size,C)

        Bwin = x.shape[0]

        qkv = self.qkv(x).reshape(Bwin,-1,3,self.num_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)

        q,k,v = qkv

        q = q * self.scale

        attn = torch.matmul(q, k.transpose(-2,-1))
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn,v)

        x = x.transpose(1,2).reshape(Bwin,-1,C)
        x = self.proj(x)

        x = x.view(
            B,
            Hp//self.window_size,
            Wp//self.window_size,
            self.window_size,
            self.window_size,
            C
        )

        x = x.permute(0,1,3,2,4,5).contiguous()
        x = x.view(B,Hp,Wp,C)

        x = x[:,:H,:W,:]

        return x.view(B,H*W,C)


# -----------------------------------------------------------
# SSTB block
# -----------------------------------------------------------

class SSTB(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.spe = SpectralMSA(dim)
        self.spa = SpatialMSA(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self,x,H,W):

        x = x + self.spe(self.norm1(x),H,W)
        x = x + self.spa(self.norm2(x),H,W)
        x = x + self.ffn(self.norm3(x))

        return x


# -----------------------------------------------------------
# Mamba Block
# -----------------------------------------------------------

class MambaBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self,x):

        B,C,H,W = x.shape

        x = x.flatten(2).transpose(1,2)

        x = self.mamba(x)

        x = x.transpose(1,2).view(B,C,H,W)

        return x


# -----------------------------------------------------------
# Main Network
# -----------------------------------------------------------

class MainNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Correct channel counts for your dataset
        num_hsi = 242
        num_msi = 4
        num_feature = 120

        self.sstb = SSTB(num_feature)

        self.mamba = MambaBlock(num_feature)

        self.x1 = nn.Parameter(torch.randn(1))
        self.x2 = nn.Parameter(torch.randn(1))

        # FIXED embedding input dimension
        self.Embedding = nn.Linear(num_hsi + num_msi, num_feature)

        self.refine = nn.Sequential(
            nn.Conv2d(num_feature,num_feature,3,1,1),
            nn.LeakyReLU(inplace=True),
            MambaBlock(num_feature),
            nn.Conv2d(num_feature,num_hsi,3,1,1)
        )


    def forward(self,HSI,MSI):

        # Upsample LR-HSI to MSI resolution
        UP_LRHSI = F.interpolate(
            HSI,
            size=MSI.shape[2:],
            mode="bicubic",
            align_corners=False
        )

        UP_LRHSI = UP_LRHSI.clamp_(0,1)

        B,C,H,W = UP_LRHSI.shape

        # Concatenate spectral + multispectral
        Data = torch.cat((UP_LRHSI,MSI),1)

        E = rearrange(Data,'B C H W -> B (H W) C')

        E = self.Embedding(E)

        E1 = self.sstb(E,H,W)

        E2 = rearrange(E,'B (H W) C -> B C H W',H=H,W=W)

        E2 = self.mamba(E2)

        E2 = rearrange(E2,'B C H W -> B (H W) C')

        E = self.x1 * E1 + self.x2 * E2

        Highpass = rearrange(E,'B (H W) C -> B C H W',H=H,W=W)

        Highpass = self.refine(Highpass)

        output = Highpass + UP_LRHSI

        output = output.clamp_(0,1)

        return output, UP_LRHSI, Highpass