"""
Hook-based cache mechanism for HunyuanVideo10.

Replaces the transformer's ``forward`` with a cache-aware version:
- On *anchor* steps (listed in ``step_list``): run ALL transformer blocks
  normally and store the intermediate (img, txt) features.
- On *non-anchor* steps: skip the heavy transformer blocks entirely and
  use the lightweight ``Predictor`` to estimate the output from the
  cached features plus the current inputs.

Usage::

    from cache_utils import apply_cache_to_transformer

    apply_cache_to_transformer(pipe.transformer)
"""

from types import MethodType
from typing import Any

import psutil
import inspect
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import loguru

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import distributed as dist

import torch
import torch.nn as nn

from typing import Any, List, Tuple, Optional, Union, Dict
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import retrieve_timesteps, rescale_noise_cfg,HunyuanVideoPipelineOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from hyvideo.constants import PRECISION_TO_TYPE
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.utils.data_utils import align_to
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
import time
from loguru import logger

EXAMPLE_DOC_STRING = """"""

def _initialize_transformer_cache(transformer):
    if not hasattr(transformer, "_cache_img_txt"):
        transformer.register_buffer("_cache_img_txt", torch.zeros(1), persistent=False)
        transformer._cache_img_txt = None


def _build_modulation_vector(transformer, t, timestep_r, text_states_2, guidance):
    vec = transformer.time_in(t)
    if transformer.enable_meanflow:
        vec = (vec + transformer.time_r_in(timestep_r)) / 2
    vec = vec + transformer.vector_in(text_states_2)

    if transformer.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + transformer.guidance_in(guidance)

    return vec


def _embed_transformer_inputs(transformer, x, t, text_states, text_mask):
    img = transformer.img_in(x)
    if transformer.text_projection == "linear":
        txt = transformer.txt_in(text_states)
    elif transformer.text_projection == "single_refiner":
        txt = transformer.txt_in(text_states, t, text_mask if transformer.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {transformer.text_projection}")

    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // transformer.patch_size[0],
        oh // transformer.patch_size[1],
        ow // transformer.patch_size[2],
    )
    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]
    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    return (
        img,
        txt,
        tt,
        th,
        tw,
        txt_seq_len,
        img_seq_len,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )


def _run_double_stream_blocks(
    transformer,
    img,
    txt,
    vec,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    freqs_cis,
):
    for block in transformer.double_blocks:
        img, txt = block(
            img,
            txt,
            vec,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cis,
        )
    return img, txt


def _run_single_stream_blocks(
    transformer,
    x,
    vec,
    txt_seq_len,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    freqs_cos,
    freqs_sin,
):
    for block in transformer.single_blocks:
        x = block(
            x,
            vec,
            txt_seq_len,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            (freqs_cos, freqs_sin),
        )
    return x


def _compute_cache_aware_states(
    transformer,
    img,
    txt,
    vec,
    txt_seq_len,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    freqs_cos,
    freqs_sin,
    cache_dic,
    predictor,
):
    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
    if cache_dic.get("step", 0) in cache_dic.get("step_list", []):
        img, txt = _run_double_stream_blocks(
            transformer,
            img,
            txt,
            vec,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cis,
        )
        x = torch.cat((img, txt), 1)
        x = _run_single_stream_blocks(
            transformer,
            x,
            vec,
            txt_seq_len,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cos,
            freqs_sin,
        )
        transformer._cache_img_txt = x.detach()
        return x

    return predictor(
        curr_latent=torch.cat((img, txt), 1),
        cache_latent=transformer._cache_img_txt,
        vec=vec,
        txt_seq_len=txt_seq_len,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        freqs_cis=(freqs_cos, freqs_sin),
    )


def _finalize_transformer_output(transformer, x, img_seq_len, vec, tt, th, tw, return_dict):
    img = x[:, :img_seq_len, ...]
    img = transformer.final_layer(img, vec)
    img = transformer.unpatchify(img, tt, th, tw)

    if return_dict:
        return {"x": img}
    return img


def _handle_deprecated_callbacks(kwargs):
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    return callback, callback_steps


def _resolve_callback_tensor_inputs(callback_on_step_end, callback_on_step_end_tensor_inputs):
    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        return callback_on_step_end.tensor_inputs
    return callback_on_step_end_tensor_inputs


def _configure_pipeline_state(
    pipe,
    guidance_scale,
    guidance_rescale,
    clip_skip,
    cross_attention_kwargs,
):
    pipe._guidance_scale = guidance_scale
    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._interrupt = False


def _get_batch_size(prompt, prompt_embeds):
    if isinstance(prompt, str):
        return 1
    if isinstance(prompt, list):
        return len(prompt)
    return prompt_embeds.shape[0]


def _get_execution_device(pipe):
    if dist.is_initialized():
        return torch.device(f"cuda:{dist.get_rank()}")
    return pipe._execution_device


def _get_lora_scale(pipe):
    if pipe.cross_attention_kwargs is None:
        return None
    return pipe.cross_attention_kwargs.get("scale", None)


def _encode_prompt_conditions(
    pipe,
    prompt,
    device,
    num_videos_per_prompt,
    negative_prompt,
    prompt_embeds,
    attention_mask,
    negative_prompt_embeds,
    negative_attention_mask,
    lora_scale,
    data_type,
):
    (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_mask,
        negative_prompt_mask,
    ) = pipe.encode_prompt(
        prompt,
        device,
        num_videos_per_prompt,
        pipe.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        attention_mask=attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_attention_mask=negative_attention_mask,
        lora_scale=lora_scale,
        clip_skip=pipe.clip_skip,
        data_type=data_type,
    )

    if pipe.text_encoder_2 is None:
        return (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
            None,
            None,
            None,
            None,
        )

    (
        prompt_embeds_2,
        negative_prompt_embeds_2,
        prompt_mask_2,
        negative_prompt_mask_2,
    ) = pipe.encode_prompt(
        prompt,
        device,
        num_videos_per_prompt,
        pipe.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=None,
        attention_mask=None,
        negative_prompt_embeds=None,
        negative_attention_mask=None,
        lora_scale=lora_scale,
        clip_skip=pipe.clip_skip,
        text_encoder=pipe.text_encoder_2,
        data_type=data_type,
    )

    return (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_mask,
        negative_prompt_mask,
        prompt_embeds_2,
        negative_prompt_embeds_2,
        prompt_mask_2,
        negative_prompt_mask_2,
    )


def _apply_classifier_free_guidance_prompts(
    pipe,
    prompt_embeds,
    negative_prompt_embeds,
    prompt_mask,
    negative_prompt_mask,
    prompt_embeds_2,
    negative_prompt_embeds_2,
    prompt_mask_2,
    negative_prompt_mask_2,
):
    if not pipe.do_classifier_free_guidance:
        return prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if prompt_mask is not None:
        prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])

    if prompt_embeds_2 is not None:
        prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        
    if prompt_mask_2 is not None:
        prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

    return prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2


def _prepare_denoising_timesteps(pipe, num_inference_steps, device, timesteps, sigmas, n_tokens):
    extra_set_timesteps_kwargs = pipe.prepare_extra_func_kwargs(
        pipe.scheduler.set_timesteps,
        {"n_tokens": n_tokens},
    )
    return retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        **extra_set_timesteps_kwargs,
    )


def _adjust_video_length_for_vae(video_length, vae_ver):
    if "884" in vae_ver:
        return (video_length - 1) // 4 + 1
    if "888" in vae_ver:
        return (video_length - 1) // 8 + 1
    return video_length


def _prepare_initial_latents(
    pipe,
    batch_size,
    num_videos_per_prompt,
    height,
    width,
    video_length,
    prompt_embeds,
    device,
    generator,
    latents,
):
    return pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        pipe.transformer.config.in_channels,
        height,
        width,
        video_length,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )


def _get_precision_settings(pipe):
    target_dtype = PRECISION_TO_TYPE[pipe.args.precision]
    autocast_enabled = (target_dtype != torch.float32) and not pipe.args.disable_autocast
    vae_dtype = PRECISION_TO_TYPE[pipe.args.vae_precision]
    vae_autocast_enabled = (vae_dtype != torch.float32) and not pipe.args.disable_autocast
    return target_dtype, autocast_enabled, vae_dtype, vae_autocast_enabled


def _get_guidance_expand(embedded_guidance_scale, latent_model_input, device, target_dtype):
    if embedded_guidance_scale is None:
        return None

    return (
        torch.tensor(
            [embedded_guidance_scale] * latent_model_input.shape[0],
            dtype=torch.float32,
            device=device,
        ).to(target_dtype)
        * 1000.0
    )


def _apply_cfg_noise(pipe, noise_pred):
    noise_pred_text = None
    if pipe.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + pipe.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    if pipe.do_classifier_free_guidance and pipe.guidance_rescale > 0.0:
        noise_pred = rescale_noise_cfg(
            noise_pred,
            noise_pred_text,
            guidance_rescale=pipe.guidance_rescale,
        )

    return noise_pred


def _handle_step_end_callback(
    callback_on_step_end,
    callback_on_step_end_tensor_inputs,
    pipe,
    step_index,
    timestep,
    latents,
    prompt_embeds,
    negative_prompt_embeds,
):
    if callback_on_step_end is None:
        return latents, prompt_embeds, negative_prompt_embeds

    available_tensors = {
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
    }
    callback_kwargs = {
        key: available_tensors[key]
        for key in callback_on_step_end_tensor_inputs
        if key in available_tensors
    }
    callback_outputs = callback_on_step_end(pipe, step_index, timestep, callback_kwargs)

    latents = callback_outputs.pop("latents", latents)
    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
    return latents, prompt_embeds, negative_prompt_embeds


def _run_denoising_loop(
    pipe,
    timesteps,
    num_inference_steps,
    latents,
    prompt_embeds,
    negative_prompt_embeds,
    prompt_mask,
    prompt_embeds_2,
    freqs_cis,
    predictor,
    step_list,
    target_dtype,
    autocast_enabled,
    device,
    embedded_guidance_scale,
    extra_step_kwargs,
    callback_on_step_end,
    callback_on_step_end_tensor_inputs,
    callback,
    callback_steps,
):
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    timesteps_prev = torch.cat([timesteps, torch.tensor([0.0], device=device)])

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            timestep_r = timesteps_prev[i + 1]
            latent_model_input = (
                torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            )
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            t_expand = t.repeat(latent_model_input.shape[0])
            timestep_r_expand = timestep_r.repeat(latent_model_input.shape[0])
            guidance_expand = _get_guidance_expand(
                embedded_guidance_scale,
                latent_model_input,
                device,
                target_dtype,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=target_dtype,
                enabled=autocast_enabled,
            ):
                noise_pred = pipe.transformer(
                    latent_model_input,
                    t_expand,
                    timestep_r=timestep_r_expand,
                    text_states=prompt_embeds,
                    text_mask=prompt_mask,
                    text_states_2=prompt_embeds_2,
                    freqs_cos=freqs_cis[0],
                    freqs_sin=freqs_cis[1],
                    guidance=guidance_expand,
                    return_dict=True,
                    cache_dic={"step": i, "step_list": step_list},
                    predictor=predictor,
                )["x"]

            noise_pred = _apply_cfg_noise(pipe, noise_pred)
            latents = pipe.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]
            latents, prompt_embeds, negative_prompt_embeds = _handle_step_end_callback(
                callback_on_step_end,
                callback_on_step_end_tensor_inputs,
                pipe,
                i,
                t,
                latents,
                prompt_embeds,
                negative_prompt_embeds,
            )

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0
            ):
                if progress_bar is not None:
                    progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(pipe.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    return latents


def _decode_latents(pipe, latents, output_type, generator, enable_tiling, vae_dtype, vae_autocast_enabled):
    if output_type == "latent":
        return latents

    expand_temporal_dim = False
    if len(latents.shape) == 4:
        if isinstance(pipe.vae, AutoencoderKLCausal3D):
            latents = latents.unsqueeze(2)
            expand_temporal_dim = True
    elif len(latents.shape) != 5:
        raise ValueError(
            f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
        )

    if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor:
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        latents = latents / pipe.vae.config.scaling_factor

    with torch.autocast(
        device_type="cuda",
        dtype=vae_dtype,
        enabled=vae_autocast_enabled,
    ):
        if enable_tiling:
            pipe.vae.enable_tiling()
        image = pipe.vae.decode(latents, return_dict=False, generator=generator)[0]

    if expand_temporal_dim or image.shape[2] == 1:
        image = image.squeeze(2)
    return image


def _finalize_pipeline_output(pipe, image, return_dict):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().float()
    pipe.maybe_free_model_hooks()

    if not return_dict:
        return image
    return HunyuanVideoPipelineOutput(videos=image)


def _resolve_seeds(seed, batch_size, num_videos_per_prompt):
    if isinstance(seed, torch.Tensor):
        seed = seed.tolist()
    if seed is None:
        return [random.randint(0, 1_000_000) for _ in range(batch_size * num_videos_per_prompt)]
    if isinstance(seed, int):
        return [
            seed + i
            for _ in range(batch_size)
            for i in range(num_videos_per_prompt)
        ]
    if isinstance(seed, (list, tuple)):
        if len(seed) == batch_size:
            return [
                int(seed[i]) + j
                for i in range(batch_size)
                for j in range(num_videos_per_prompt)
            ]
        if len(seed) == batch_size * num_videos_per_prompt:
            return [int(s) for s in seed]
        raise ValueError(
            f"Length of seed must be equal to number of prompt(batch_size) or "
            f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
        )
    raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")


def _validate_predict_dimensions(height, width, video_length):
    if width <= 0 or height <= 0 or video_length <= 0:
        raise ValueError(
            f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
        )
    if (video_length - 1) % 4 != 0:
        raise ValueError(f"`video_length-1` must be a multiple of 4, got {video_length}")


def _prepare_predict_prompts(sampler, prompt, negative_prompt, guidance_scale):
    if not isinstance(prompt, str):
        raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
    prompt = [prompt.strip()]

    if negative_prompt is None or negative_prompt == "":
        negative_prompt = sampler.default_negative_prompt
    if guidance_scale == 1.0:
        negative_prompt = ""
    if not isinstance(negative_prompt, str):
        raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")

    return prompt, [negative_prompt.strip()]


def _configure_predict_scheduler(sampler, flow_shift):
    sampler.pipeline.scheduler = FlowMatchDiscreteScheduler(
        shift=flow_shift,
        reverse=sampler.args.flow_reverse,
        solver=sampler.args.flow_solver,
    )


def _build_predict_debug_string(
    target_height,
    target_width,
    target_video_length,
    prompt,
    negative_prompt,
    seed,
    infer_steps,
    num_videos_per_prompt,
    guidance_scale,
    n_tokens,
    flow_shift,
    embedded_guidance_scale,
):
    return f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}"""


def apply_disca(sampler):
    """Monkey-patch *transformer* with a cache-aware forward method.

    After calling this function the model exposes:
    - ``transformer.cache_img`` / ``transformer.cache_txt``: cached features.
    - The ``forward`` method accepts extra kwargs ``cache_dic`` and ``predictor``.
    """

    _initialize_transformer_cache(sampler.pipeline.transformer)

    def new_cache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        timestep_r: torch.Tensor = None,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
        cache_dic: dict | None = None,
        predictor: nn.Module | None = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        vec = _build_modulation_vector(self, t, timestep_r, text_states_2, guidance)
        (
            img,
            txt,
            tt,
            th,
            tw,
            txt_seq_len,
            img_seq_len,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ) = _embed_transformer_inputs(self, x, t, text_states, text_mask)
        x = _compute_cache_aware_states(
            self,
            img,
            txt,
            vec,
            txt_seq_len,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cos,
            freqs_sin,
            cache_dic or {},
            predictor,
        )
        return _finalize_transformer_output(
            self,
            x,
            img_seq_len,
            vec,
            tt,
            th,
            tw,
            return_dict,
        )



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call_disca__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        data_type: str = "video",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        predictor: nn.Module | None = None,
        step_list: List[int] = [0, 1, 2, 5, 9, 13, 17, 19],
        **kwargs,
    ):

        callback, callback_steps = _handle_deprecated_callbacks(kwargs)
        callback_on_step_end_tensor_inputs = _resolve_callback_tensor_inputs(
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
        )
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )
        _configure_pipeline_state(
            self,
            guidance_scale,
            guidance_rescale,
            clip_skip,
            cross_attention_kwargs,
        )

        batch_size = _get_batch_size(prompt, prompt_embeds)
        device = _get_execution_device(self)
        lora_scale = _get_lora_scale(self)
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = _encode_prompt_conditions(
            self,
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
            prompt_embeds,
            attention_mask,
            negative_prompt_embeds,
            negative_attention_mask,
            lora_scale,
            data_type,
        )
        prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2 = (
            _apply_classifier_free_guidance_prompts(
                self,
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                negative_prompt_mask,
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            )
        )
        timesteps, num_inference_steps = _prepare_denoising_timesteps(
            self,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            n_tokens,
        )
        video_length = _adjust_video_length_for_vae(video_length, vae_ver)
        latents = _prepare_initial_latents(
            self,
            batch_size,
            num_videos_per_prompt,
            height,
            width,
            video_length,
            prompt_embeds,
            device,
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )
        target_dtype, autocast_enabled, vae_dtype, vae_autocast_enabled = _get_precision_settings(self)
        latents = _run_denoising_loop(
            self,
            timesteps,
            num_inference_steps,
            latents,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            prompt_embeds_2,
            freqs_cis,
            predictor,
            step_list,
            target_dtype,
            autocast_enabled,
            device,
            embedded_guidance_scale,
            extra_step_kwargs,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            callback,
            callback_steps,
        )
        image = _decode_latents(
            self,
            latents,
            output_type,
            generator,
            enable_tiling,
            vae_dtype,
            vae_autocast_enabled,
        )
        return _finalize_pipeline_output(self, image, return_dict)

    @torch.no_grad()
    def predict_disca(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6.0,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        predictor=None,
        step_list=None,
        **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                infer_steps (int): The number of inference steps. Default is 100.
        """
        out_dict = {}
        seeds = _resolve_seeds(seed, batch_size, num_videos_per_prompt)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

        _validate_predict_dimensions(height, width, video_length)
        logger.info(f"Input (height, width, video_length) = ({height}, {width}, {video_length})")
        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length
        out_dict["size"] = (target_height, target_width, target_video_length)

        prompt, negative_prompt = _prepare_predict_prompts(
            self,
            prompt,
            negative_prompt,
            guidance_scale,
        )
        _configure_predict_scheduler(self, flow_shift)
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length,
            target_height,
            target_width,
        )
        n_tokens = freqs_cos.shape[0]
        logger.debug(
            _build_predict_debug_string(
                target_height,
                target_width,
                target_video_length,
                prompt,
                negative_prompt,
                seed,
                infer_steps,
                num_videos_per_prompt,
                guidance_scale,
                n_tokens,
                flow_shift,
                embedded_guidance_scale,
            )
        )

        start_time = time.time()

        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            num_videos_per_prompt=num_videos_per_prompt,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            predictor=predictor,
            step_list=step_list,
            is_progress_bar=True,
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict

    sampler.predict_disca = MethodType(predict_disca, sampler)

    sampler.pipeline.__class__.__call__ = __call_disca__

    sampler.pipeline.transformer.forward = MethodType(new_cache_forward, sampler.pipeline.transformer)
