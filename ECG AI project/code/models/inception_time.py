# models/inception_time.py
import torch, torch.nn as nn

class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=32, ks=(9,19,39), use_res=True):
        super().__init__()
        self.use_res = use_res
        if bottleneck and in_ch > 1:
            self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False); bch = bottleneck
        else:
            self.bottleneck = nn.Identity(); bch = in_ch
        self.conv_k = nn.ModuleList([nn.Conv1d(bch, out_ch, k, padding=k//2, bias=False) for k in ks])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch*(len(ks)+1))
        self.act = nn.ReLU(inplace=True)
        if self.use_res:
            self.res = nn.Sequential(
                nn.Conv1d(in_ch, out_ch*(len(ks)+1), 1, bias=False),
                nn.BatchNorm1d(out_ch*(len(ks)+1))
            )
    def forward(self, x):
        x_in = x
        x = self.bottleneck(x)
        y = [c(x) for c in self.conv_k]
        y.append(self.conv_pool(self.maxpool(x_in)))
        y = self.bn(torch.cat(y, 1))
        return self.act(y + self.res(x_in)) if self.use_res else self.act(y)

class InceptionTime(nn.Module):
    def __init__(self, c_in, c_out, n_blocks=6, out_ch=32, bottleneck=32, ks=(9,19,39), dropout=0.1):
        super().__init__()
        layers, in_ch = [], c_in
        for _ in range(n_blocks):
            layers.append(InceptionBlock1D(in_ch, out_ch, bottleneck=bottleneck, ks=ks, use_res=True))
            in_ch = out_ch*(len(ks)+1)
        self.feature = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, c_out)
    def forward(self, x):  # x:[B,C,L]
        x = self.feature(x)
        x = self.gap(x).squeeze(-1)
        x = self.drop(x)
        return self.head(x)
