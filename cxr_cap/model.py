"""
Paper ka multimodal model (Fig. 2, 3, 4 + Eq. 5, 6):

  CXR  -> modified ResNet-50 -> FC(2048->128) -> LeakyReLU -> BN        => x_c (B,128)
  Lab  -> 5x[FC -> LeakyReLU -> BN] (N->8N->4N->2N->N->N) -> Conv(->32) -> BN  => x_l (B,32)
  Text -> Embedding -> 2-layer Bi-GRU -> FC -> LeakyReLU -> BN          => x_p (B,32)
  concat(x_c, x_l, x_p) -> FC(->32) -> LeakyReLU -> FC(->2) -> softmax

PediCXR mein sirf CXR hai, isliye use_lab=False, use_text=False se ye model paper ke
"Chest X-Ray (ResNet-50)" baseline (Table 2, row 1) ban jata hai.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50, ResNet50_Weights


def fc_block(i, o):
    # Eq (5): y = BatchNorm(LeakyReLU(Wx + b)), LeakyReLU slope = 0.01
    return nn.Sequential(nn.Linear(i, o), nn.LeakyReLU(0.01), nn.BatchNorm1d(o))


class CXRBranch(nn.Module):
    def __init__(self, out_dim=128, pretrained=True):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # conv1(7x7)+BN+ReLU+MaxPool, layer1..4 (3,4,6,3 bottlenecks), AvgPool -> (B,2048,1,1)
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
        self.pool = net.avgpool
        self.decoder = fc_block(2048, out_dim)  # Feature decoder

    def forward(self, x):
        f = self.pool(self.backbone(x)).flatten(1)
        return self.decoder(f)


class LabBranch(nn.Module):
    def __init__(self, n_indicators=8, out_dim=32):
        super().__init__()
        N = n_indicators
        dims = [N, 8 * N, 4 * N, 2 * N, N, N]            # 5 FC layers, length har layer par half
        self.fcs = nn.Sequential(*[fc_block(dims[i], dims[i + 1]) for i in range(5)])
        self.proj = nn.Conv1d(N, out_dim, kernel_size=1)  # paper: "convolution layer" -> 32
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):                                 # x: (B, N), already normalized/imputed
        h = self.fcs(x).unsqueeze(-1)                     # (B,N,1)
        return self.bn(self.proj(h).squeeze(-1))


class TextBranch(nn.Module):
    # Paper: vocab 642 words, fixed length 27, 0 = empty/padding, 2-layer Bi-GRU, output 32.
    # embed_dim / hidden paper mein nahi diye -> ye MERI assumption hai (64 / 64).
    def __init__(self, vocab_size=643, embed_dim=64, hidden=64, out_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.head = fc_block(2 * hidden, out_dim)

    def forward(self, x):                                 # x: (B,27) long
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                         # (4,B,H): [l1 fwd, l1 bwd, l2 fwd, l2 bwd]
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)          # last layer, dono directions
        return self.head(h)


class MultimodalCAP(nn.Module):
    def __init__(self, use_cxr=True, use_lab=False, use_text=False,
                 n_indicators=8, vocab_size=643, pretrained=True):
        super().__init__()
        assert use_cxr or use_lab or use_text
        self.use_cxr, self.use_lab, self.use_text = use_cxr, use_lab, use_text
        self.cxr = CXRBranch(128, pretrained) if use_cxr else None
        self.lab = LabBranch(n_indicators, 32) if use_lab else None
        self.text = TextBranch(vocab_size, out_dim=32) if use_text else None
        fused = 128 * use_cxr + 32 * use_lab + 32 * use_text   # full model: 128+32+32 = 192
        self.classifier = nn.Sequential(nn.Linear(fused, 32), nn.LeakyReLU(0.01), nn.Linear(32, 2))

    def forward(self, cxr=None, lab=None, text=None):
        feats = []
        if self.use_cxr:
            feats.append(self.cxr(cxr))
        if self.use_lab:
            feats.append(self.lab(lab))
        if self.use_text:
            feats.append(self.text(text))
        return self.classifier(torch.cat(feats, dim=1))   # logits; softmax = F.softmax(.,1)

    @property
    def cam_layer(self):
        """GradCAM ke liye last conv block."""
        return self.cxr.backbone[-1][-1]
