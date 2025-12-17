import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------- Initialization ----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# --------------------- Spectral MSA -------------------------------------------
class SpectralMSA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.alpha = nn.Parameter(torch.ones(num_heads))
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H, W):
        # x: (B, N, C), N = H*W
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv

        # (B, heads, head_dim, N)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        outs = []
        for i in range(self.num_heads):
            attn = torch.matmul(k[:, i], q[:, i].transpose(-2, -1))
            attn = attn / (self.alpha[i] + 1e-6)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            o = torch.matmul(attn, v[:, i]).transpose(-2, -1)
            outs.append(o)

        x = torch.cat(outs, dim=-1)  # (B, N, C)
        x = self.proj(x)
        return x


# --------------------- Spatial MSA -------------------------------------------
class SpatialMSA(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, H, W):
        # x: (B, N, C)
        B, N, C = x.shape
        x_windows = x.view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_windows = F.pad(x_windows, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x_windows.size(1), x_windows.size(2)

        # partition into windows
        x_windows = x_windows.view(
            B,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
            C,
        )
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        Bwin = x_windows.shape[0]
        qkv = self.qkv(x_windows).reshape(Bwin, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # (3, Bwin, heads, tokens, head_dim)

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x_attn = torch.matmul(attn, v).transpose(1, 2).reshape(Bwin, -1, C)
        x_attn = self.proj(x_attn)

        x = x_attn.view(
            B,
            Hp // self.window_size,
            Wp // self.window_size,
            self.window_size,
            self.window_size,
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        x = x[:, :H, :W, :].contiguous()
        return x.view(B, H * W, C)


# --------------------- SSTB Block (HMFormer) ---------------------------------
class SSTB(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.spe = SpectralMSA(dim, num_heads, dropout)
        self.spa = SpatialMSA(dim, num_heads, window_size, dropout)

        mid = int(dim * mlp_ratio)  # IMPORTANT: cast to int

        self.ffn = nn.Sequential(
            nn.Linear(dim, mid),
            nn.GELU(),
            nn.Linear(mid, dim)
        )

    def forward(self, x, H, W):
        x = x + self.spe(self.norm1(x), H, W)
        x = x + self.spa(self.norm2(x), H, W)
        x = x + self.ffn(self.norm3(x))
        return x


# --------------------- Generic Transformer Blocks ----------------------------
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
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48,
                 sp_sz=64*64, num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim,
                                                  dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # If you want positional embedding:
        # x = x + self.pos_embedding[:, :x.size(1), :]
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer_D(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48,
                 sp_sz=64*64, num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim,
                                                  dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # If you want positional embedding:
        # x = x + self.pos_embedding[:, :x.size(1), :]
        for attn1, attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, mask=mask)
            x = ff(x)
        return x


# --------------------- MainNet (HMFormer + MDAS) -----------------------------
class MainNet(nn.Module):
    def __init__(self, C_hsi=31, C_msi=3, embed_dim=48):
        super(MainNet, self).__init__()

        self.C_hsi = C_hsi
        self.C_msi = C_msi
        self.embed_dim = embed_dim

        # Embedding from concatenated HSI + MSI
        self.Embedding = nn.Linear(C_hsi + C_msi, embed_dim)

        # HMFormer backbone
        self.sstb = SSTB(
            dim=embed_dim,
            num_heads=4,
            window_size=8,
            mlp_ratio=4.0,
            dropout=0.1,
        )

        # MDAS transformer bottleneck
        self.T_E = Transformer_E(embed_dim)
        self.T_D = Transformer_D(embed_dim)

        # HMFormer hierarchical fusion
        self.conv_1x1_1 = nn.Conv1d(2 * embed_dim, embed_dim, 1)
        self.conv_1x1_2 = nn.Conv1d(2 * embed_dim, embed_dim, 1)
        self.conv_1x1_3 = nn.Conv1d(2 * embed_dim, embed_dim, 1)

        # Final refinement to HSI bands
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(embed_dim, C_hsi, 3, 1, 1),   # [B, C_hsi, H, W]
        )

        init_weights(self.Embedding, self.refine)

    def forward(self, HSI, MSI):
        """
        HSI: [B, C_hsi, H_l, W_l]  (LR)
        MSI: [B, C_msi, H_h, W_h]  (HR)
        """

        # 1) Upsample HSI to MSI resolution if needed
        B, _, H_msi, W_msi = MSI.shape
        if HSI.shape[2:] != (H_msi, W_msi):
            UP = F.interpolate(
                HSI, size=(H_msi, W_msi),
                mode='bicubic', align_corners=False
            )
        else:
            UP = HSI

        UP = UP.clamp_(0, 1)                      # [B, C_hsi, H, W]
        _, _, H, W = UP.shape

        # 2) Fuse & embed
        fused = torch.cat((UP, MSI), dim=1)       # [B, C_hsi+C_msi, H, W]
        X = rearrange(fused, 'B C H W -> B (H W) C')
        X0 = self.Embedding(X)                    # [B, HW, embed_dim]

        # 3) HMFormer SSTB hierarchy
        X1 = self.sstb(X0, H, W)
        X2 = self.sstb(X1, H, W)
        X3 = self.sstb(X2, H, W)
        X4 = self.sstb(X3, H, W)

        # 4) MDAS transformer bottleneck on deepest features
        Xe = self.T_E(X4)                         # [B, HW, embed_dim]
        Xe = self.T_D(Xe)                         # [B, HW, embed_dim]

        # 5) HMFormer-style hierarchical fusion
        s13 = torch.cat((Xe, X3), dim=-1)         # [B, HW, 2*embed_dim]
        s13_t = s13.transpose(1, 2)               # [B, 2*embed_dim, HW]
        S1 = self.conv_1x1_1(s13_t).transpose(1, 2)

        s12 = torch.cat((S1, X2), dim=-1)
        s12_t = s12.transpose(1, 2)
        S2 = self.conv_1x1_2(s12_t).transpose(1, 2)

        s11 = torch.cat((S2, X1), dim=-1)
        s11_t = s11.transpose(1, 2)
        S3 = self.conv_1x1_3(s11_t).transpose(1, 2)

        # 6) Refine & reconstruct
        Highpass = rearrange(S3, "B (H W) C -> B C H W", H=H, W=W)
        Highpass = self.refine(Highpass)          # [B, C_hsi, H, W]

        # 7) Residual super-resolution
        out = (Highpass + UP).clamp_(0, 1)

        return out
