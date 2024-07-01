# Copyright 2023-2024 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as audio_transforms
from einops import rearrange
from einops.layers.torch import Rearrange
from loguru import logger
from torch.cuda.amp import autocast
from transformers import PretrainedConfig, PreTrainedModel

from models.ced.layers import (AudioPatchEmbed, DropPath, Mlp, drop_patches,
                               to_2tuple, trunc_normal_)


class FrontEnd(nn.Sequential):

    def __init__(
        self,
        f_min: int = 0,
        sample_rate: int = 16000,
        win_size: int = 512,
        center: bool = True,
        n_fft: int = 512,
        f_max: Optional[int] = None,
        hop_size: int = 160,
        n_mels: int = 64,
    ):
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size
        self.n_mels = n_mels

        super().__init__(
            audio_transforms.MelSpectrogram(
                f_min=self.f_min,
                sample_rate=self.sample_rate,
                win_length=self.win_size,
                center=self.center,
                n_fft=self.n_fft,
                f_max=self.f_max,
                hop_length=self.hop_size,
                n_mels=self.n_mels,
            ),
            audio_transforms.AmplitudeToDB(top_db=120),
        )

    # Disable Autocast for FP16 training!
    @autocast(enabled=False)
    def forward(self, x):
        return super().forward(x)


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # ========== Changed by Li ==========
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # ========== Changed by Li ==========

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
        #                           C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(
        #     0)  # make torchscript happy (cannot use tensor as tuple)

        # ========== Changed by Li ==========
        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        # ========== Changed by Li ==========

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if mask is not None:
        # # Mask is a tensor of shape [B, T, T]
        # # Different from self.causal == True, the mask might be something like:
        # # [False, False, True]
        # # [False, False, True]
        # # [True, True, True]
        # # We use -inf to pad here, since if we would pad by any number, the entries at rows only containing
        # # [True, True, True] would lead to weights such as: [0.33,0.33,0.33], which is not correct
        # mask_value = torch.as_tensor(-float('inf'))
        # print(mask.shape, attn.shape)
        # attn = attn.masked_fill(mask, mask_value)
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)
        attn = attn.softmax(dim=-1)
        # Only for the case that a mask with all True entries on a row is passed.
        # attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = Attention,
        attention_kwargs={},
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            **attention_kwargs,
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CEDConfig(PretrainedConfig):
    def __init__(self, cfg=None):
        super().__init__()
        self.pad_last = True
        self.embed_dim = 1280

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class AudioTransformer(PreTrainedModel):

    def __init__(
        self,
        cfg,
        outputdim=527,
        patch_size=16,
        patch_stride=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_bn: bool = True,
        norm_layer=None,
        act_layer=None,
        init_values=None,
        target_length=1012,
        pooling="mean",
        wavtransforms=None,
        spectransforms=None,
        time_patch_out: Optional[float] = None,
        freq_patch_out: Optional[float] = None,
        block_type=Block,
        attention_type=Attention,
        eval_avg="mean",
        **kwargs,
    ):
        super().__init__(cfg)
        self.cfg = cfg
        assert pooling in ("mean", "token", "dm", "logit")
        self.outputdim = outputdim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = kwargs.get("n_mels", 64)
        n_fft = kwargs.get("n_fft", 512)
        self.hop_size = kwargs.get("hop_size", 160)
        self.win_size = kwargs.get("win_size", 512)
        f_min = kwargs.get("f_min", 0)
        f_max = kwargs.get("f_max", 8000)
        self.center = kwargs.get("center", True)
        self.pad_last = kwargs.get("pad_last", True)
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out

        self.front_end = FrontEnd(
            f_min=f_min,
            f_max=f_max,
            center=self.center,
            win_size=self.win_size,
            hop_size=self.hop_size,
            sample_rate=16000,
            n_fft=n_fft,
            n_mels=self.n_mels,
        )

        self.init_bn = nn.Sequential(
            Rearrange("b c f t -> b f c t"),
            torch.nn.BatchNorm2d(self.n_mels, momentum=0.01),
            Rearrange("b f c t -> b c f t"),
        )
        self.target_length = target_length

        patch_stride = to_2tuple(self.patch_stride)[-1]
        # Allowed length in number of frames, otherwise the positional embedding will throw an error
        self.maximal_allowed_length = self.target_length

        self.patch_embed = AudioPatchEmbed(
            input_size=(self.n_mels, target_length),
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            flatten=False,
            patch_stride=self.patch_stride,
        )
        self.spectransforms = (
            nn.Sequential() if spectransforms is None else spectransforms
        )
        self.wavtransforms = nn.Sequential() if wavtransforms is None else wavtransforms

        if self.pooling == "token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        self.time_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * 0.02
        )
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * 0.02
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(
            *[
                block_type(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attention_type=attention_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.outputlayer = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, outputdim)
        )
        self.apply(self.init_weights)
        if hasattr(self, "cls_token"):
            nn.init.normal_(self.cls_token, std=1e-6)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"time_pos_embed", "cls_token", "freq_pos_embed", "token_pos_embed"}

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = (
            x + self.freq_pos_embed[:, :, :, :]
        )  # Just to support __getitem__ in posembed

        if self.training and self.time_patch_out is not None:
            x = drop_patches(x, dim=-1, frac=self.time_patch_out)
        if self.training and self.freq_patch_out is not None:
            x = drop_patches(x, dim=-2, frac=self.freq_patch_out)

        x = rearrange(x, "b c f t -> b (f t) c")
        if self.pooling == "token":
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "token":
            x = x[:, 0]
            return self.outputlayer(x).sigmoid()
        elif self.pooling == "mean":
            x = x.mean(1)
            return self.outputlayer(x).sigmoid()
        elif self.pooling == "logit":
            x = x.mean(1)
            return self.outputlayer(x)
        elif self.pooling == "dm":
            # Unpack using the frequency dimension, which is constant
            x = rearrange(x, "b (f t) d -> b f t d", f=self.patch_embed.grid_size[0])
            # First poolin frequency, then sigmoid the (B T D) output
            x = self.outputlayer(x.mean(1)).sigmoid()
            return x.mean(1)
        else:
            return x.mean(1)

    def load_state_dict(self, state_dict, strict=True):
        if (
            "time_pos_embed" in state_dict
            and hasattr(self, "time_pos_embed")
            and self.time_pos_embed.shape != state_dict["time_pos_embed"].shape
        ):
            logger.debug(
                "Positional Embedding shape not the same with model, resizing!"
            )
            self.change_pos_embedding(state_dict)
        if any("qkv" in i for i in state_dict.keys()):
            self.split_qkv(state_dict)
        super().load_state_dict(state_dict, strict=strict)

    def split_qkv(self, state_dict):
        updates = {}
        removes = []
        for i in state_dict.keys():
            if "qkv" in i:
                data = state_dict[i]
                qkv_size = data.shape[0] // 3
                updates[i.replace("qkv", "q_proj")] = data[:qkv_size]
                updates[i.replace("qkv", "k_proj")] = data[qkv_size : qkv_size * 2]
                updates[i.replace("qkv", "v_proj")] = data[qkv_size * 2 :]
                removes.append(i)
        state_dict.update(updates)
        for rm in removes:
            del state_dict[rm]

    def change_pos_embedding(self, state_dict):
        target_time_pos_embed_length = self.time_pos_embed.shape[-1]
        target_freq_pos_embed_length = self.freq_pos_embed.shape[-2]

        pretrained_time_pos_embed = state_dict["time_pos_embed"]
        pretrained_freq_pos_embed = state_dict["freq_pos_embed"]

        if target_time_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
            state_dict["time_pos_embed"] = pretrained_time_pos_embed[
                ..., :target_time_pos_embed_length
            ]
        else:
            state_dict["time_pos_embed"] = torch.nn.functional.interpolate(
                pretrained_time_pos_embed,
                size=(1, target_time_pos_embed_length),
                align_corners=False,
                mode="bilinear",
            )
        if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-2]:
            state_dict["freq_pos_embed"] = pretrained_freq_pos_embed[
                :, :, :target_freq_pos_embed_length, :
            ]
        else:
            state_dict["freq_pos_embed"] = torch.nn.functional.interpolate(
                pretrained_freq_pos_embed,
                size=(target_freq_pos_embed_length, 1),
                align_corners=False,
                mode="bilinear",
            )

    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b f t -> b 1 f t")
        x = self.init_bn(x)
        if x.shape[-1] > self.maximal_allowed_length:
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(
                        *x.shape[:-1], self.target_length, device=x.device
                    )
                    pad[..., : splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = rearrange(splits, "spl b c f t-> (spl b) c f t")

            # ========== Changed by Li ==========
            # x = self.forward_head(self.forward_features(x))
            x = self.forward_features(x)
            # x = rearrange(x, '(spl b) d -> spl b d', spl=n_splits)

            # --- Approach 1 ---
            x = rearrange(x, "(spl b) t d -> spl b t d", spl=n_splits)
            x = rearrange(x, "spl b t d -> b (spl t) d")
            # --- Approach 2 ---
            # x = x.view(n_splits, x.shape[0] // n_splits, x.shape[-2], x.shape[-1])
            # x = x.permute(1, 0, 2, 3).contiguous()
            # x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

            # if self.eval_avg == 'mean':
            #     x = x.mean(0)
            # elif self.eval_avg == 'max':
            #     x = x.max(0)[0]
            # else:
            #     raise ValueError(f'Unknown Eval average function ({self.eval_avg})')
            # ========== Changed by Li ==========

        else:
            x = self.forward_features(x)
            # x = self.forward_head(x)        # Changed by Li
        return x

    def forward(self, x):
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        if self.training:
            x = self.spectransforms(x)
        x = self.forward_spectrogram(x)
        return x
