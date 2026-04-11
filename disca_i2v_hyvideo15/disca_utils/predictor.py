from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import MMDoubleStreamBlock, MMSingleStreamBlock
import torch
from hyvideo.models.transformers.modules.activation_layers import get_activation_layer
from typing import Optional
import torch.nn as nn
from omegaconf import ListConfig

class Predictor(nn.Module):
    def __init__(
        self, 
        model_depth: int,
        hidden_size: int = 2048,
        heads_num: int = 16,
        attn_mode: str = "flash",
        attn_kwargs: Optional[dict] = None,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        moe_config=None,
    ):

        super().__init__()
        self.model_depth = model_depth
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.mlp_width_ratio = mlp_width_ratio
        self.mlp_act_type = mlp_act_type
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type
        self.attn_mode = attn_mode

        mm_single_blocks_depth = 0

        if not isinstance(attn_mode, (list, tuple, ListConfig)):
            attn_mode = [attn_mode] * self.model_depth
        else:
            assert len(attn_mode) == self.model_depth

        factory_kwargs = {"device": device, "dtype": dtype}

        self.fusion_mlp_img = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            get_activation_layer("silu")(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.fusion_mlp_txt = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            get_activation_layer("silu")(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        double_blocks = []

        for d_block_index in range(self.model_depth):
            double_blocks.append(
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=self.mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode[d_block_index],
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    dtype=dtype,
                    device=device,
                )
            )
        self.double_blocks = nn.ModuleList(double_blocks)

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode[s_block_index + self.model_depth],
                    attn_kwargs=attn_kwargs,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    moe_config=moe_config,
                    **factory_kwargs,
                )
                for s_block_index in range(mm_single_blocks_depth)
            ]
        )

    def forward(self, 
                img, txt,
                cache_img, cache_txt,
                vec, freqs_cis, text_mask,
                attn_param):

        target_device = img.device
        if next(self.parameters()).device != target_device:
            self.to(target_device)

        img = torch.cat([img, cache_img], dim=-1)
        img = self.fusion_mlp_img(img)

        txt = torch.cat([txt, cache_txt], dim=-1)
        txt = self.fusion_mlp_txt(txt)

        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and attn_param["win_type"] == "hybrid"
                and attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            attn_param["layer-name"] = f"double_block_{index+1}"
            img,txt = block(img=img,
                            txt=txt,
                            vec=vec,
                            freqs_cis=freqs_cis,
                            text_mask=text_mask,
                            attn_param=attn_param,
                            is_flash=force_full_attn,
                            block_idx=index)


        return img,txt
