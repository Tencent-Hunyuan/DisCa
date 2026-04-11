from hyvideo.modules.models import MMSingleStreamBlock
import torch.nn as nn
import torch
from hyvideo.modules.activation_layers import get_activation_layer
from typing import Optional

class Predictor(nn.Module):
    def __init__(
        self, 
        model_depth: int = 2,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.model_depth = model_depth
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.mlp_width_ratio = mlp_width_ratio
        self.mlp_act_type = mlp_act_type
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            get_activation_layer("silu")(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.mm_single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=self.mlp_width_ratio,
                    mlp_act_type=self.mlp_act_type,
                    qk_norm=self.qk_norm,
                    qk_norm_type=self.qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(self.model_depth)
            ]
        )

    def forward(
        self,
        curr_latent,
        cache_latent,
        vec,
        txt_seq_len,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        freqs_cis,
    ):

        x = torch.cat([curr_latent, cache_latent], dim=-1)
        x = self.fusion_mlp(x)

        for layer, block in enumerate(self.mm_single_blocks):
            predictor_args = [
                x,
                vec,
                txt_seq_len,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]
            x = block(*predictor_args)

        return x
