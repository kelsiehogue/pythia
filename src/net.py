import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if self.bias is not None and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return qk_dots + self.bias

class StyleConv3d(torch.nn.Module):
    def __init__(self, inchans, outchans, kernel_size, style_size, eps=1e-8):
        super(StyleConv3d, self).__init__()
        self.eps = eps
        self.inchans = inchans
        self.outchans = outchans
        self.kernel_size = kernel_size
        self.style_size = style_size

        self.weight = torch.nn.Parameter(torch.randn(outchans, inchans, kernel_size[0], kernel_size[1], 1))
        self.style_linear_scale = torch.nn.Linear(self.style_size, self.inchans)
        self.noise_param = torch.nn.Parameter(torch.zeros(outchans, 1, 1, 1))
        self.bias_param = torch.nn.Parameter(torch.zeros(outchans, 1, 1, 1))
       

    def forward(self, x, style):
        b, c, h, w, t = x.shape
        if style is not None:
            style = self.style_linear_scale(style)
        else:
            x = F.conv3d(x, self.weight, padding='same')
            x = x + (self.noise_param * torch.randn(b, 1, h, w, t, device=x.device))
            x = x + self.bias_param
            return x


        style = style[:, None, :, None, None, None]
        weight = self.weight[None, :, :, :, :, :]
        weight = weight * style
        sigma_inv = torch.rsqrt((weight ** 2).sum(dim=(2,3,4,5), keepdim=True) + 1e-8)

        weight = weight * sigma_inv

        x = x.reshape(1, -1, h, w, t)
        _, _, *w_shape = weight.shape
        weight = weight.reshape(b * self.outchans, *w_shape)

        x = F.conv3d(x, weight, padding='same', groups=b)

        x = x.reshape(-1, self.outchans, h, w, t)

        x = x + (self.noise_param * torch.randn(b, 1, h, w, t, device=x.device))
        x = x + self.bias_param

        return x

# class StyleLinear(torch.nn.Module):
#     def __init__(self, indim, outdim, style_size):
#         super(StyleLinear, self).__init__()
#         self.indim = indim
#         self.outdim = outdim
#         self.style_size = style_size

#         self.weight = torch.nn.Parameter(torch.randn(self.outdim, self.indim))
#         self.style_linear = torch.nn.Linear(self.style_size, self.indim)
#         self.bias_param = torch.nn.Parameter(torch.zeros(self.outdim))
#     def forward(self, x, style):
#         if style is not None:
#             style = self.style_linear(style)
#         else:
#             x = F.linear(x, self.weight, bias=None)
#             x = x + self.bias_param
#             return x
        
#         weight = self.weight * style
#         sigma_inv = torch.rsqrt((weight ** 2).sum(dim=(1), keepdim=True) + 1e-8)
#         weight = weight * sigma_inv

#         x = F.linear(x, weight, bias=None)
#         x = x + self.bias_param
#         return x

# class Patchify(torch.nn.Module):
#     def __init__(self, patch_size):
#         super(Patchify, self).__init__()
#         self.patch_size = patch_size
#     def forward(self, x):
#         B,C,H,W,T = x.shape
#         patches = []
#         for r in range(T):
#             x_splice = x[:, :, :, :, r]
#             splice_patch = F.unfold(x_splice, self.patch_size, stride=self.patch_size)
#             patches.append(splice_patch)
#         patches = torch.cat(patches, dim=-1).transpose(1, 2)
#         return patches, x.shape

# class UnPatchify(torch.nn.Module):
#     def __init__(self, patch_size):
#         super(UnPatchify, self).__init__()
#         self.patch_size = patch_size
#     def forward(self, x, size):
#         patches = torch.chunk(x.transpose(2, 1), size[-1], dim=-1)

#         x_images = []
#         for p in patches:
#             x_images.append(F.fold(p, (size[2], size[3]), kernel_size=self.patch_size, stride=self.patch_size))
#         x_images = torch.stack(x_images, dim=-1)
#         return x_images

# class Attention(torch.nn.Module):
#     def __init__(self, dim, num_heads, full_attention=False):
#         super(Attention, self).__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.full_attention = full_attention

#         self.alibi = AlibiPositionalBias(num_heads)
#     def forward(self, q, k, v):
#         B, N, C = q.shape

#         q = q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
#         k = k.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
#         v = v.view(B, N, self.num_heads, self.head_dim).transpose(1,2)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

#         attn_scores = self.alibi(attn_scores)

#         if not self.full_attention:
#             attn_mask = torch.tril(torch.ones_like(attn_scores)).bool()
#             attn_scores = attn_scores.masked_fill_(mask=attn_mask, value=-1e18)

#         attn = torch.softmax(attn_scores, dim=-1)
#         attn = torch.matmul(attn, v)

#         out = attn.transpose(2,1).contiguous().view(B, N, C)
#         return out

# class SelfAttention(torch.nn.Module):
#     def __init__(self, dim, num_heads, style_size):
#         super(SelfAttention, self).__init__()
#         self.dim = dim
#         self.num_heads = num_heads

#         self.qkv_linear = StyleLinear(self.dim, self.dim*3, style_size)
#         self.out_linear = StyleLinear(self.dim, self.dim, style_size)
#         self.attention = Attention(dim, num_heads)
#     def forward(self, x, style):
#         qkv = self.qkv_linear(x, style)
#         q,k,v = torch.chunk(qkv, 3, -1)
#         out = self.attention(q,k,v)
#         out = self.out_linear(out, style)
#         return out

# class FFN(torch.nn.Module):
#     def __init__(self, dim, expansion, style_size):
#         super(FFN, self).__init__()
#         self.dim = dim
#         self.expansion = expansion
#         self.style_size = style_size

#         self.linear1 = StyleLinear(self.dim, self.dim, style_size)
#         self.linear2 = StyleLinear(self.dim, self.dim, style_size)
#         self.activation = torch.nn.GELU()
#     def forward(self, x, style):
#         x = self.linear1(x, style)
#         x = self.activation(x)
#         x = self.linear2(x, style)
#         return x

# class Block(torch.nn.Module):
#     def __init__(self, d_model, nheads, expansion, style_size):
#         super(Block, self).__init__()
#         self.nheads = nheads
#         self.expansion = expansion

#         self.attn = SelfAttention(d_model, self.nheads, style_size)
#         self.ffn1 = FFN(d_model, self.expansion, style_size)
#         self.layer_norm1 = torch.nn.LayerNorm((d_model,))
#         self.layer_norm2 = torch.nn.LayerNorm((d_model,))
#     def forward(self, x, latent):
#         if torch.rand(1) > .5 or not self.training:
#             y = self.attn(x, latent)
#         else:
#             y = torch.zeros_like(x)
#         x = y + x
#         x = self.layer_norm1(x)

#         if torch.rand(1) > .5 or not self.training:
#             y = self.ffn1(x, latent)
#         else:
#             y = torch.zeros_like(x)
#         x = y + x
#         x = self.layer_norm2(x)

#         return x


class FrameAttention(torch.nn.Module):
    def __init__(self, nheads, inchans, kernel_size, style_size, autoregressive_masking=True):
        super(FrameAttention, self).__init__()
        self.nheads = nheads
        self.kernel_size = kernel_size
        self.inchans = inchans
        self.style_size = style_size
        self.autoregressive_masking = autoregressive_masking
        self.qkv_conv = StyleConv3d(inchans, inchans*3, (1,1), style_size=style_size)
        self.proj_conv = StyleConv3d(inchans, inchans, (1,1), style_size=style_size)
        self.alibi = AlibiPositionalBias(nheads)

    def forward(self, x, style):
        qkv = self.qkv_conv(x, style) 

        q,k,v = torch.chunk(qkv, 3, 1)
        q_shape = q.shape

        q = torch.flatten(q, 1, -2).transpose(1, 2)
        k = torch.flatten(k, 1, -2).transpose(1, 2)
        v = torch.flatten(v, 1, -2).transpose(1, 2)

        q = torch.stack(q.chunk(self.nheads, -1), 1)
        k = torch.stack(k.chunk(self.nheads, -1), 1)
        v = torch.stack(v.chunk(self.nheads, -1), 1)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

        attn_scores = self.alibi(attn)
        attn_mask = torch.tril(torch.ones_like(attn_scores)).bool()
        attn_scores = attn_scores.masked_fill_(mask=attn_mask, value=-1e18)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = torch.matmul(attn, v)
        attn = attn.transpose(1,2)
        attn = attn.flatten(2,-1)
        attn = attn.unflatten(-1, q_shape[1:4])
        attn = attn.permute(0, 2, 3, 4, 1)
        attn = self.proj_conv(attn, style)

        return attn
        

class FrameFFN(torch.nn.Module):
    def __init__(self, expansion, inchans, kernel_size, style_size):
        super(FrameFFN, self).__init__()
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.inchans = inchans
        self.conv1 = StyleConv3d(inchans, inchans*expansion, (1,1), style_size=style_size)
        self.conv2 = StyleConv3d(inchans*expansion, inchans*expansion, (kernel_size[0], kernel_size[1]), style_size=style_size)
        self.conv3 = StyleConv3d(inchans*expansion, inchans, (1,1), style_size=style_size)
        self.swish = nn.GELU()
    def forward(self, x, style):
        x = self.conv1(x, style)
        x = self.swish(x)
        x = self.conv2(x, style)
        x = self.swish(x)
        x = self.conv3(x, style)
        return x

class FrameBlock(torch.nn.Module):
    def __init__(self, nheads, expansion, inchans, kernel_size, autoregressive_masking=True, style_size=1024):
        super(FrameBlock, self).__init__()

        self.nheads = nheads
        self.expansion = expansion
        self.inchans = inchans
        self.kernel_size = kernel_size
        self.attn = FrameAttention(self.nheads, self.inchans, self.kernel_size, style_size, autoregressive_masking)
        self.ffn = FrameFFN(self.expansion, self.inchans, self.kernel_size, style_size)
        self.swish = torch.nn.GELU()
    def forward(self, x, latent):
        y = self.attn(x, latent) + x
        y = self.ffn(y, latent) + x
        y = self.swish(y)
        return x

class StyleFFN(torch.nn.Module):
    def __init__(self, insize, nlayers=8):
        super(StyleFFN, self).__init__()
        self.insize = insize
        self.nlayers = nlayers
        self.linears = torch.nn.ModuleList([torch.nn.Linear(insize, insize) for i in range(nlayers)])
        self.activations = torch.nn.ModuleList([torch.nn.ReLU() for i in range(nlayers)])
    def forward(self, latent):
        for i, j in zip(self.linears, self.activations):
            latent = i(latent)
            latent = j(latent)
        return latent


class PythianEngine(torch.nn.Module):
    def __init__(self,  nheads,
                        expansion,
                        nlayers,
                        inchans=3,
                        outchans=3,
                        kernel_size=(7,7),
                        patch_size=(8,8),
                        style_size=1024,
                        autoregressive_masking=True,
                        d_model=768
                        ):

        super(PythianEngine, self).__init__()
        self.nheads = nheads
        self.expansion = expansion
        self.nlayers = nlayers
        self.inchans = inchans
        self.outchans = outchans
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.style_size = style_size
        self.autoregressive_masking = autoregressive_masking

        self.stylization = StyleFFN(style_size)

        self.layers = torch.nn.ModuleList([FrameBlock(nheads, expansion, d_model, kernel_size, style_size=style_size) for i in range(self.nlayers)])
        self.inconvs = torch.nn.Conv3d(inchans, d_model, 1)
        self.outconvs = torch.nn.Conv3d(d_model, inchans, 1)
    def forward(self, x, latent=None):

        latent = self.stylization(latent) if latent != None else None

        x = self.inconvs(x)
        for i in range(self.nlayers):
            x = torch.utils.checkpoint.checkpoint(self.layers[i], x, latent)
        x = self.outconvs(x)

        return x