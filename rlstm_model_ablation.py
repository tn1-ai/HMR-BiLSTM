"""
rlstm_model_ablation.py
========================
Drop-in replacement cho rlstm_model.py dùng cho ablation study.
Thêm 3 flag vào RLSTMCell và RLSTMClassifier, KHÔNG đụng gì đến code gốc.

Cách dùng:
    from rlstm_model_ablation import RLSTMClassifier, RLSTMLoss

    # Full model (giống rlstm_model.py gốc)
    model = RLSTMClassifier(use_rmc=True, use_cnn=True, use_attention=True)

    # Ablation No-RMC: c_t = c_lstm (bỏ toàn bộ RMC path)
    model = RLSTMClassifier(use_rmc=False)

    # Ablation No-CNN: raw ECG đi thẳng vào BiRLSTM
    model = RLSTMClassifier(use_cnn=False)

    # Ablation Mean-Pool: thay AttentionPooling bằng mean
    model = RLSTMClassifier(use_attention=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  CNN FEATURE EXTRACTOR (không đổi)
# =============================================================================
class ECGFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_channels=64, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, output_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(output_channels)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        return x.transpose(1, 2)


# =============================================================================
#  ATTENTION POOLING (không đổi)
# =============================================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, h_seq):
        scores  = self.attention(h_seq)
        weights = F.softmax(scores, dim=1)
        pooled  = (h_seq * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)


# =============================================================================
#  HMR-BiLSTM CELL — với flag use_rmc
# =============================================================================
class RLSTMCell(nn.Module):
    """
    HMR-BiLSTM cell với ablation flag use_rmc.

    use_rmc=True  (default): full Hybrid Memory Path
        c_t = beta * c_rmc + (1-beta) * c_lstm

    use_rmc=False (No-RMC ablation): bỏ toàn bộ RMC path
        c_t = c_lstm  (tương đương BiLSTM thuần với CNN + Attention)
        Các weight W_c, W_h_rmc, W_alpha, W_beta vẫn được tạo nhưng
        không được dùng trong forward → tránh thay đổi param count một
        cách đột ngột gây so sánh không fair. (Nếu muốn fair param count
        thực sự, dùng use_rmc=False kết hợp với exclude_rmc_params=True
        trong optimizer — nhưng với ablation đơn giản thì không cần.)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 dropout: float = 0.1, use_rmc: bool = True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.use_rmc     = use_rmc

        # LSTM gates (luôn có)
        self.W_x = nn.Linear(input_size,  4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)

        # RMC components — chỉ dùng khi use_rmc=True
        # Vẫn tạo để checkpoint compatible giữa variants
        self.W_c     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h_rmc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.W_alpha    = nn.Linear(2 * hidden_size, 2)
        self.W_beta     = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.last_r_t    = None
        self.last_c_keep = None
        self.last_c_add  = None
        self.last_alpha  = None
        self.last_beta   = None
        self.last_c_rmc  = None
        self.last_c_lstm = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if any(k in name for k in ["W_h", "W_c", "W_h_rmc"]):
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "W_x" in name or "W_h" in name:
                    n = param.size(0) // 4
                    with torch.no_grad():
                        param[n:2*n].fill_(1.0)
                if "W_beta.bias" in name:
                    with torch.no_grad():
                        param.fill_(-5.0)

    def forward(self, x_t, h_prev, c_prev):
        # === Bước 1: LSTM gates ===
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_size, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        # === LSTM path (luôn tính) ===
        c_lstm = f_t * c_prev + i_t * g_t

        if self.use_rmc:
            # === Bước 2: Interaction term ===
            Wc_c = self.W_c(c_prev)
            Wh_h = self.W_h_rmc(h_prev)
            m_t  = Wc_c * Wh_h

            # === Bước 3: Residual gate ===
            r_t = torch.sigmoid(self.layer_norm(Wc_c + Wh_h + m_t))

            # === Bước 4: Memory decomposition ===
            c_keep = (f_t + r_t * (1.0 - f_t)) * c_prev
            c_add  = (1.0 - r_t) * (i_t * g_t)

            # === Bước 5a: RMC path — softmax attention ===
            c_keep_n = F.layer_norm(c_keep, (c_keep.shape[-1],))
            c_add_n  = F.layer_norm(c_add,  (c_add.shape[-1],))
            scores   = self.W_alpha(torch.cat([c_keep_n, c_add_n], dim=-1))
            alpha    = F.softmax(scores, dim=-1)
            c_rmc    = alpha[:, 0:1] * c_keep + alpha[:, 1:2] * c_add

            # === Bước 5c: Blend gate ===
            beta = torch.sigmoid(self.W_beta(h_prev))

            # === Bước 5d: Hybrid output ===
            c_t = beta * c_rmc + (1.0 - beta) * c_lstm

            # Lưu cho interpretability
            self.last_r_t    = r_t
            self.last_c_keep = c_keep
            self.last_c_add  = c_add
            self.last_alpha  = alpha
            self.last_beta   = beta
            self.last_c_rmc  = c_rmc
            self.last_c_lstm = c_lstm

        else:
            # No-RMC: c_t = c_lstm thuần túy
            # Lưu placeholder để RLSTMLayer không crash
            zeros = torch.zeros_like(c_lstm)
            self.last_r_t    = zeros
            self.last_c_keep = zeros
            self.last_c_add  = zeros
            self.last_alpha  = None
            self.last_beta   = None
            self.last_c_rmc  = None
            self.last_c_lstm = c_lstm
            c_t = c_lstm

        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


# =============================================================================
#  HMR-BiLSTM LAYER (không đổi interface, chỉ pass use_rmc xuống cell)
# =============================================================================
class RLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, use_rmc=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RLSTMCell(input_size, hidden_size,
                              dropout=dropout, use_rmc=use_rmc)

    def forward(self, x, h0=None, c0=None):
        B, T, _ = x.size()
        device  = x.device
        h = torch.zeros(B, self.hidden_size, device=device) if h0 is None else h0
        c = torch.zeros(B, self.hidden_size, device=device) if c0 is None else c0

        h_outputs, r_outputs, ck_outputs, ca_outputs = [], [], [], []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            h_outputs.append(h)
            r_outputs.append(self.cell.last_r_t.clone())
            ck_outputs.append(self.cell.last_c_keep.clone())
            ca_outputs.append(self.cell.last_c_add.clone())

        return (
            torch.stack(h_outputs,  dim=1),
            (h, c),
            torch.stack(r_outputs,  dim=1),
            torch.stack(ck_outputs, dim=1),
            torch.stack(ca_outputs, dim=1),
        )


# =============================================================================
#  BIDIRECTIONAL HMR-BiLSTM
# =============================================================================
class BiRLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, dropout=0.1, use_rmc=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = input_size if i == 0 else 2 * hidden_size
            self.fwd_layers.append(
                RLSTMLayer(layer_in, hidden_size, dropout=dropout, use_rmc=use_rmc)
            )
            self.bwd_layers.append(
                RLSTMLayer(layer_in, hidden_size, dropout=dropout, use_rmc=use_rmc)
            )

        self.inter_dropout = nn.Dropout(dropout) if num_layers > 1 else None

    def forward(self, x):
        h_seq = x
        for i in range(self.num_layers):
            h_fwd, (h_T_fwd, c_T_fwd), r_fwd, ck_fwd, ca_fwd = \
                self.fwd_layers[i](h_seq)

            x_rev  = torch.flip(h_seq, dims=[1])
            h_bwd, (h_T_bwd, c_T_bwd), r_bwd, ck_bwd, ca_bwd = \
                self.bwd_layers[i](x_rev)

            h_bwd  = torch.flip(h_bwd,  dims=[1])
            r_bwd  = torch.flip(r_bwd,  dims=[1])
            ck_bwd = torch.flip(ck_bwd, dims=[1])
            ca_bwd = torch.flip(ca_bwd, dims=[1])

            h_seq = torch.cat([h_fwd, h_bwd], dim=-1)
            if self.inter_dropout is not None and i < self.num_layers - 1:
                h_seq = self.inter_dropout(h_seq)

        h_T = torch.cat([h_T_fwd, h_T_bwd], dim=-1)
        c_T = torch.cat([c_T_fwd, c_T_bwd], dim=-1)
        return h_seq, (h_T, c_T), r_fwd, r_bwd, ck_fwd, ck_bwd, ca_fwd, ca_bwd


# =============================================================================
#  CLASSIFIER — 3 ablation flags
# =============================================================================
class RLSTMClassifier(nn.Module):
    """
    RLSTMClassifier với 3 ablation flags:

    use_rmc=True/False       → full hybrid memory vs LSTM path only
    use_cnn=True/False       → CNN feature extractor vs raw input
    use_attention=True/False → attention pooling vs mean pooling
    """

    def __init__(
        self,
        input_size:      int   = 1,
        hidden_size:     int   = 96,
        dropout:         float = 0.25,
        num_classes:     int   = 5,
        cnn_out_channels: int  = 64,
        num_layers:      int   = 1,
        use_rmc:         bool  = True,
        use_cnn:         bool  = True,
        use_attention:   bool  = True,
    ):
        super().__init__()
        self.use_rmc       = use_rmc
        self.use_cnn       = use_cnn
        self.use_attention = use_attention

        # CNN — Ablation No-CNN: bỏ, BiRLSTM nhận raw input (input_size=1)
        if use_cnn:
            self.cnn       = ECGFeatureExtractor(
                input_channels=input_size,
                output_channels=cnn_out_channels,
                dropout=dropout * 0.5,
            )
            birlstm_input  = cnn_out_channels
        else:
            self.cnn       = None
            birlstm_input  = input_size   # raw ECG: 1

        # BiRLSTM — pass use_rmc flag xuống đến từng cell
        self.birlstm = BiRLSTM(
            birlstm_input, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_rmc=use_rmc,
        )

        # Pooling — Ablation Mean-Pool: thay bằng mean over time axis
        if use_attention:
            self.attention_pool = AttentionPooling(2 * hidden_size)
        else:
            self.attention_pool = None

        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, return_internals=False):
        # CNN hoặc raw
        if self.cnn is not None:
            features = self.cnn(x)
        else:
            features = x   # (B, T, 1) đi thẳng vào BiRLSTM

        outputs = self.birlstm(features)
        h_seq   = outputs[0]
        r_fwd, r_bwd = outputs[2], outputs[3]

        # Pooling
        if self.attention_pool is not None:
            h_pooled, attn_weights = self.attention_pool(h_seq)
        else:
            # Mean pooling: trung bình theo chiều thời gian
            h_pooled     = h_seq.mean(dim=1)
            attn_weights = None

        h_pooled = self.layer_norm(h_pooled)
        h_pooled = self.dropout(h_pooled)
        logits   = self.classifier(h_pooled)

        if return_internals:
            internals = {
                "r_fwd":             r_fwd,
                "r_bwd":             r_bwd,
                "c_keep_fwd":        outputs[4],
                "c_keep_bwd":        outputs[5],
                "c_add_fwd":         outputs[6],
                "c_add_bwd":         outputs[7],
                "attention_weights": attn_weights,
            }
            return logits, internals
        return logits


# =============================================================================
#  LOSS (copy nguyên từ rlstm_model.py để file này tự đủ)
# =============================================================================
def temporal_smoothness_loss(r_seq):
    if r_seq.size(1) < 2:
        return torch.tensor(0.0, device=r_seq.device, requires_grad=True)
    diff = r_seq[:, 1:, :] - r_seq[:, :-1, :]
    return (diff ** 2).sum(dim=-1).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_raw = F.cross_entropy(logits, targets, reduction="none")
        pt     = torch.exp(-ce_raw)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_raw
        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[targets]
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class RLSTMLoss(nn.Module):
    def __init__(self, lambda_smooth=0.01, class_weights=None,
                 use_focal=False, focal_gamma=2.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        if use_focal:
            self.task_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.task_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets, r_fwd=None, r_bwd=None):
        l_task   = self.task_loss(logits, targets)
        l_smooth = torch.tensor(0.0, device=logits.device)
        if r_fwd is not None:
            l_smooth = l_smooth + temporal_smoothness_loss(r_fwd)
        if r_bwd is not None:
            l_smooth = l_smooth + temporal_smoothness_loss(r_bwd)
        if r_fwd is not None and r_bwd is not None:
            l_smooth = l_smooth / 2.0
        total = l_task + self.lambda_smooth * l_smooth
        return total, {
            "task":   l_task.detach(),
            "smooth": l_smooth.detach(),
            "total":  total.detach(),
        }
