
import torch
from torch import nn
from torch.nn.utils import weight_norm
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from mamba_ssm import Mamba  # 需要安装 mamba-ssm 包

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

#加上tcn + 不选残差

class TeLU(nn.Module):
    """
    TeLU activation: x * tanh(exp(x))
    为了数值稳定性，对 exp 的输入做了截断（clamp）。
    clamp_range 可调，通常取 [-20, 20] 已足够安全。
    """
    def __init__(self, clamp_min: float = -20.0, clamp_max: float = 20.0):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 截断后计算，避免 exp 导致的溢出或 inf
        x_clamped = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x * torch.tanh(torch.exp(x_clamped))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.act = TeLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class STA(nn.Module):
    def __init__(self, temporal_kernel , heads ,dropout, alpha):
        super().__init__()
        self.dropout = nn.Dropout(alpha * dropout)

        self.cnn_low = weight_norm(nn.Conv2d(heads, heads, (3, 1),
                              stride=1, padding=self.get_padding(3)))


        self.cnn_high = weight_norm(nn.Conv2d(heads, heads, (temporal_kernel, 1),
                                             stride=1, padding=self.get_padding(temporal_kernel)))

        self.cnn_fuse = weight_norm(nn.Conv2d(2*heads, heads, (1, 1),(1, 1)))

        self.bn = nn.BatchNorm2d(heads)

    def forward(self, x):
        x = self.dropout(x) #(16,16,408,16)
        x1 = self.cnn_low(x)

        return x1
    def get_padding(self, kernel):
        return (int(0.5 * (kernel - 1)), 0)

class Attention(nn.Module):
    def __init__(self, dim, temporal_kernel, heads = 8, dim_head = 64, anchor=3, dropout = 0., alpha=0.25):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.STA = STA(temporal_kernel,  heads, dropout, alpha)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = self.STA(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Frequency(nn.Module):
    #等我们的论文被接受，我们就公开完整代码
    #Once our paper is accepted, we will release the complete code.

'''简单版：加入简单频率'''
class TTransformer(nn.Module):
    def __init__(self, dim, temporal_kernel, depth, heads, dim_head, mlp_dim,
                 n_fft=64, hop_length=32, win_length=64, dropout=0.1, alpha=0.25):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.freqbranch = Frequency(n_fft, hop_length, win_length)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, temporal_kernel, heads=heads, dim_head=dim_head,
                                       dropout=dropout, alpha=alpha)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, Mamba(d_model=dim_head, d_state=dim_head, d_conv=4, expand=2))
            ]))
        self.to(DEVICE)
    def forward(self, x):
        x = x.to(DEVICE) # x=()
        stack_time =[]
        open = False
        # x: [batch, seq_len, dim]
        for attn, ff, mamba in self.layers:
            if(open):
                # 时间维度下采样
                x = F.max_pool1d(x.transpose(1,2), kernel_size=2, padding=0).transpose(1,2)
            # 在下采样序列上做注意力
            x_attn = attn(x) + x
            # 原始分支：Mamba 及频谱特征
            x_mamba = mamba(x)
            x_branch = self.freqbranch(x_mamba.transpose(1,2)).transpose(1,2)
            # 融合所有分支
            x = x_attn + x_mamba + x_branch
            x = ff(x) + x
            stack_time.append(x)
            open = True
        x = torch.cat(stack_time, dim= 1) #(1,286,32)
        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class  BHTMNet(nn.Module):


    def __init__(self, *, num_chan, num_time, layers_transformer, hidden_graph, num_head, alpha, temporal_kernel, num_kernel=64,
                 num_classes, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__()

        self.cnn_encoder_time = Conv2dWithConstraint(1, num_kernel, kernel_size =(1, temporal_kernel), padding=self.get_padding((1, temporal_kernel)[-1]), max_norm=2)

        self.cnn_encoder_global = Conv2dWithConstraint(num_kernel, num_kernel, (num_chan, 1), padding=0, max_norm=2)

        self.cnn_encoder_half = Conv2dWithConstraint(num_kernel, num_kernel, (num_chan//2, 1), padding=0, max_norm=2)

        #Dropout避免过拟合，因为Conv1d会拟合（Conv2dWithConstraint会约束权重，所以不会拟合，但会欠拟合）
        self.tcn = nn.Sequential(
            nn.Conv1d(num_kernel, num_kernel, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(num_kernel), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(num_kernel, num_kernel, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(num_kernel), nn.ELU(), nn.Dropout(dropout)
        )

        self.bn = nn.BatchNorm1d(num_kernel)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(56, stride=56)

        #如何判断是否为时间，就看模型输入数据的位置，这里num_time对应dim
        dim = int(0.5*num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        # 因为 x 数据的形状为(batch, num_kernel, dim)，所以输入pos_embedding中为(1, num_kernel, dim)

        self.ok = 321  # 每次都要修改

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, self.ok))

        self.proj = nn.Linear(hidden_graph, dim_head)
        self.transformer = TTransformer(
            temporal_kernel=temporal_kernel,
            depth=layers_transformer,
            dim=dim_head, heads=num_head,
            dim_head=dim_head, dropout=dropout, mlp_dim=dim_head,
            alpha=alpha
        )

        self.MLP = nn.Sequential(
            nn.Linear(dim_head, num_classes)
        )

        self.mamba = Mamba(
            d_model=dim_head,
            d_state=dim_head,
            d_conv=4,
            expand=2,
        )
        self.fuse_proj = nn.Linear(2 * dim_head, dim_head)
        self.fuse_ln = nn.LayerNorm(dim_head)
        self.fuse_dropout = nn.Dropout(dropout)
        self.to(DEVICE)


    def forward(self, eeg):
        eeg = eeg.to(DEVICE)
        # eeg: (b, chan, time) = (16,32,1000)
        eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time) = (16,1,32,1000)
    #在进入 Transformer 之前对原始 EEG 信号进行初步的时序和空间特征提取
        # x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time) = (16,64,1,500)   若没有空间卷积则数据为（16,64,32,500）

        x = self.cnn_encoder_time(eeg) #(16, 64, 32, 1000)

        x1 = self.cnn_encoder_global(x) #(16, 64, 1, 1000)

        x2 = self.cnn_encoder_half(x) #(16, 64, 11, 1000)

        x = torch.cat((x1, x2), dim=2)#(16, 64, 12, 1000)

        x = self.to_patch_embedding(x)

        x = self.tcn(x)


        x = self.bn(x)

        x = self.act(x)

        x = self.pool(x)

        b, n, _ = x.shape #(16,64,816)

        #x += self.pos_embedding 的作用是把位置（顺序）信息加入到每个 token 的向量表示里（它告诉 Transformer “这个 token 在序列中的位置/身份是什么）
        x += self.pos_embedding # (16,64,500)
        x = x.permute(0, 2, 1)

        x_trans = self.proj(x) #(16,408,16)
        x_trans = self.transformer(x_trans)  # (16,408,64)

        x = torch.mean(x_trans, dim=-2)
        x = self.MLP(x)
        return x

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data = torch.ones((16, 32, 1000))
    emt = BHTMNet(num_chan=32, num_time=1000, layers_transformer = 1, hidden_graph = 64, num_head = 16, alpha = 0.25, temporal_kernel=11, num_kernel=64,
                 num_classes=2, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.5)
    print(emt)
    print(count_parameters(emt))

    out = emt(data)
