"""
Hook-based cache mechanism for HunyuanVideo15Transformer3DModel (diffusers).

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
from typing import Any, Dict, List, Optional, Union

import loguru

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import distributed as dist


from hyvideo.commons import (
    auto_offload_model,
    get_rank,
)
from hyvideo.commons.parallel_states import get_parallel_state

from hyvideo.pipelines.pipeline_utils import retrieve_timesteps, rescale_noise_cfg
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideoPipelineOutput

import torch
import torch.nn as nn

from hyvideo.utils.communications import all_gather
from typing import Any, List, Tuple, Optional, Union, Dict
from hyvideo.commons.parallel_states import get_parallel_state

def _initialize_transformer_cache(transformer):
    if not hasattr(transformer, "_cache_img"):
        transformer.register_buffer("_cache_img", torch.zeros(1), persistent=False)
        transformer._cache_img = None
    if not hasattr(transformer, "_cache_txt"):
        transformer.register_buffer("_cache_txt", torch.zeros(1), persistent=False)
        transformer._cache_txt = None


def _get_default_guidance(hidden_states, guidance):
    if guidance is not None:
        return guidance
    return torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)


def _prepare_transformer_inputs(
    transformer,
    hidden_states,
    timestep,
    encoder_attention_mask,
    freqs_cos,
    freqs_sin,
):
    img = hidden_states
    text_mask = encoder_attention_mask
    _, _, ot, oh, ow = hidden_states.shape
    tt, th, tw = (
        ot // transformer.patch_size[0],
        oh // transformer.patch_size[1],
        ow // transformer.patch_size[2],
    )
    transformer.attn_param["thw"] = [tt, th, tw]

    if freqs_cos is None and freqs_sin is None:
        freqs_cos, freqs_sin = transformer.get_rotary_pos_embed((tt, th, tw))

    img = transformer.img_in(img)
    parallel_dims = get_parallel_state()
    if parallel_dims.sp_enabled:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
        if img.shape[1] % sp_size != 0:
            n_token = img.shape[1]
            assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f"Too short context length for SP {sp_size}"
        img = torch.chunk(img, sp_size, dim=1)[sp_rank]
        freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
        freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

    return img, text_mask, timestep, tt, th, tw, freqs_cos, freqs_sin, parallel_dims


def _build_modulation_vector(transformer, timestep, text_states_2, guidance, timestep_r):
    vec = transformer.time_in(timestep)

    if text_states_2 is not None:
        vec = vec + transformer.vector_in(text_states_2)

    if transformer.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + transformer.guidance_in(guidance)

    if timestep_r is not None:
        vec = vec + transformer.time_r_in(timestep_r)

    return vec


def _embed_text_states(transformer, text_states, timestep, text_mask):
    if transformer.text_projection == "linear":
        txt = transformer.txt_in(text_states)
    elif transformer.text_projection == "single_refiner":
        txt = transformer.txt_in(
            text_states,
            timestep,
            text_mask if transformer.use_attention_mask else None,
        )
    else:
        raise NotImplementedError(f"Unsupported text_projection: {transformer.text_projection}")

    if transformer.cond_type_embedding is not None:
        cond_emb = transformer.cond_type_embedding(
            torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
        )
        txt = txt + cond_emb

    return txt


def _append_byt5_tokens(transformer, txt, text_mask, extra_kwargs):
    if not transformer.glyph_byT5_v2:
        return txt, text_mask

    byt5_text_states = extra_kwargs["byt5_text_states"]
    byt5_text_mask = extra_kwargs["byt5_text_mask"]
    byt5_txt = transformer.byt5_in(byt5_text_states)
    if transformer.cond_type_embedding is not None:
        cond_emb = transformer.cond_type_embedding(
            torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
        )
        byt5_txt = byt5_txt + cond_emb

    return transformer.reorder_txt_token(
        byt5_txt,
        txt,
        byt5_text_mask,
        text_mask,
        zero_feat=True,
    )


def _append_vision_tokens(transformer, txt, text_mask, vision_states, mask_type):
    if transformer.vision_in is None or vision_states is None:
        return txt, text_mask

    bs = vision_states.shape[0]
    extra_encoder_hidden_states = transformer.vision_in(vision_states)
    if mask_type == "t2v" and torch.all(vision_states == 0):
        extra_attention_mask = torch.zeros(
            (bs, extra_encoder_hidden_states.shape[1]),
            dtype=text_mask.dtype,
            device=text_mask.device,
        )
        extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
    else:
        extra_attention_mask = torch.ones(
            (bs, extra_encoder_hidden_states.shape[1]),
            dtype=text_mask.dtype,
            device=text_mask.device,
        )

    if transformer.cond_type_embedding is not None:
        cond_emb = transformer.cond_type_embedding(
            2
            * torch.ones_like(
                extra_encoder_hidden_states[:, :, 0],
                dtype=torch.long,
                device=extra_encoder_hidden_states.device,
            )
        )
        extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

    return transformer.reorder_txt_token(
        extra_encoder_hidden_states,
        txt,
        extra_attention_mask,
        text_mask,
    )


def _force_full_attention(transformer, block_index, total_blocks):
    return (
        transformer.attn_mode in ["flex-block-attn"]
        and transformer.attn_param["win_type"] == "hybrid"
        and transformer.attn_param["win_ratio"] > 0
        and (
            (block_index + 1) % transformer.attn_param["win_ratio"] == 0
            or (block_index + 1) == total_blocks
        )
    )


def _run_double_stream_blocks(transformer, img, txt, vec, freqs_cis, text_mask):
    for index, block in enumerate(transformer.double_blocks):
        transformer.attn_param["layer-name"] = f"double_block_{index+1}"
        img, txt = block(
            img=img,
            txt=txt,
            vec=vec,
            freqs_cis=freqs_cis,
            text_mask=text_mask,
            attn_param=transformer.attn_param,
            is_flash=_force_full_attention(transformer, index, len(transformer.double_blocks)),
            block_idx=index,
        )
    return img, txt


def _run_single_stream_blocks(
    transformer,
    img,
    txt,
    vec,
    freqs_cos,
    freqs_sin,
    text_mask,
    output_features,
    output_features_stride,
):
    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]
    transformer._cache_img = img.detach()
    transformer._cache_txt = txt.detach()

    x = torch.cat((img, txt), 1)
    features_list = [] if output_features else None
    for index, block in enumerate(transformer.single_blocks):
        transformer.attn_param["layer-name"] = f"single_block_{index+1}"
        x = block(
            x=x,
            vec=vec,
            txt_len=txt_seq_len,
            freqs_cis=(freqs_cos, freqs_sin),
            text_mask=text_mask,
            attn_param=transformer.attn_param,
            is_flash=_force_full_attention(transformer, index, len(transformer.single_blocks)),
        )
        if output_features and index % output_features_stride == 0:
            features_list.append(x[:, :img_seq_len, ...])

    return x[:, :img_seq_len, ...], features_list


def _compute_cache_aware_hidden_states(
    transformer,
    img,
    txt,
    vec,
    freqs_cos,
    freqs_sin,
    text_mask,
    cache_dic,
    predictor,
    output_features,
    output_features_stride,
):
    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
    step = cache_dic.get("step", 0)
    step_list = cache_dic.get("step_list", [])

    if step in step_list:
        img, txt = _run_double_stream_blocks(transformer, img, txt, vec, freqs_cis, text_mask)
        return _run_single_stream_blocks(
            transformer,
            img,
            txt,
            vec,
            freqs_cos,
            freqs_sin,
            text_mask,
            output_features,
            output_features_stride,
        )

    img, _ = predictor(
        img=img,
        txt=txt,
        cache_img=transformer._cache_img,
        cache_txt=transformer._cache_txt,
        vec=vec,
        freqs_cis=freqs_cis,
        text_mask=text_mask,
        attn_param=transformer.attn_param,
    )
    return img, None


def _finalize_transformer_outputs(
    transformer,
    img,
    vec,
    parallel_dims,
    tt,
    th,
    tw,
    output_features,
    features_list,
):
    img = transformer.final_layer(img, vec)
    if parallel_dims.sp_enabled:
        img = all_gather(img, dim=1, group=parallel_dims.sp_group)
    img = transformer.unpatchify(img, tt, th, tw)

    if not output_features:
        return img, None

    features_list = torch.stack(features_list, dim=0)
    if parallel_dims.sp_enabled:
        features_list = all_gather(features_list, dim=2, group=parallel_dims.sp_group)
    return img, features_list


def _resolve_generation_defaults(pipe, guidance_scale, embedded_guidance_scale, flow_shift, num_inference_steps):
    guidance_scale = pipe.config.guidance_scale if guidance_scale is None else guidance_scale
    embedded_guidance_scale = (
        pipe.config.embedded_guidance_scale
        if embedded_guidance_scale is None
        else embedded_guidance_scale
    )
    flow_shift = pipe.config.flow_shift if flow_shift is None else flow_shift
    num_inference_steps = (
        pipe.config.num_inference_steps if num_inference_steps is None else num_inference_steps
    )
    return guidance_scale, embedded_guidance_scale, flow_shift, num_inference_steps


def _validate_embedded_guidance(pipe, embedded_guidance_scale):
    if embedded_guidance_scale is not None:
        assert not pipe.do_classifier_free_guidance
        assert pipe.transformer.config.guidance_embed
    else:
        assert not pipe.transformer.config.guidance_embed


def _prepare_task_inputs(reference_image):
    if reference_image is None:
        return "t2v", None, None

    if isinstance(reference_image, str):
        reference_image = Image.open(reference_image).convert("RGB")
    elif not isinstance(reference_image, Image.Image):
        raise ValueError("reference_image must be a PIL Image or path to image file")

    return "i2v", reference_image, np.array(reference_image)


def _broadcast_sp_object(value):
    obj_list = [value]
    parallel_state = get_parallel_state()
    group_src_rank = dist.get_global_rank(parallel_state.sp_group, 0)
    dist.broadcast_object_list(obj_list, src=group_src_rank, group=parallel_state.sp_group)
    return obj_list[0]


def _maybe_rewrite_prompt(pipe, prompt_rewrite, prompt, reference_image, task_type):
    if not prompt_rewrite:
        return prompt

    from hyvideo.utils.rewrite.rewrite_utils import run_prompt_rewrite

    rewritten_prompt = prompt
    if not dist.is_initialized() or get_parallel_state().sp_rank == 0:
        try:
            rewritten_prompt = run_prompt_rewrite(prompt, reference_image, task_type)
        except Exception as exc:
            loguru.logger.warning(f"Failed to rewrite prompt: {exc}")
            rewritten_prompt = prompt

    if dist.is_initialized() and get_parallel_state().sp_enabled:
        rewritten_prompt = _broadcast_sp_object(rewritten_prompt)

    return rewritten_prompt


def _configure_scheduler(pipe, flow_shift):
    pipe.scheduler = pipe._create_scheduler(
        pipe.config.flow_shift if flow_shift is None else flow_shift
    )


def _resolve_seed_and_generator(pipe, seed, generator):
    if seed is None or seed == -1:
        seed = random.randint(100000, 999999)

    if get_parallel_state().sp_enabled:
        assert seed is not None
        if dist.is_initialized():
            seed = _broadcast_sp_object(seed)

    if generator is None and seed is not None:
        generator = torch.Generator(device=pipe.noise_init_device).manual_seed(seed)

    return seed, generator


def _resolve_generation_resolution(pipe, reference_image, aspect_ratio, target_resolution):
    if reference_image is not None:
        if pipe.ideal_resolution is not None and target_resolution != pipe.ideal_resolution:
            raise ValueError(
                f"The loaded pipeline is trained for {pipe.ideal_resolution} resolution, but received input for {target_resolution} resolution. "
            )
        return pipe.get_closest_resolution_given_reference_image(reference_image, target_resolution)

    if pipe.ideal_resolution is None:
        raise ValueError("ideal_resolution is not set")
    if ":" not in aspect_ratio:
        raise ValueError("aspect_ratio must be separated by a colon")

    width, height = aspect_ratio.split(":")
    if not width.isdigit() or not height.isdigit() or int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive integers and separated by a colon in aspect_ratio")

    return pipe.get_closest_resolution_given_original_size(
        (int(width), int(height)),
        pipe.ideal_resolution,
    )


def _get_batch_size(prompt):
    if isinstance(prompt, str):
        return 1
    if isinstance(prompt, list):
        return len(prompt)
    return 1


def _log_generation_summary(
    pipe,
    user_prompt,
    prompt,
    prompt_rewrite,
    aspect_ratio,
    task_type,
    width,
    height,
    video_length,
    user_reference_image,
    reference_image,
    guidance_scale,
    embedded_guidance_scale,
    flow_shift,
    seed,
    num_inference_steps,
):
    if get_rank() != 0:
        return

    print(
        "\n"
        f"{'=' * 60}\n"
        f"🎬  HunyuanVideo Generation Task\n"
        f"{'-' * 60}\n"
        f"User Prompt:               {user_prompt}\n"
        f"Rewritten Prompt:          {prompt if prompt_rewrite else '<disabled>'}\n"
        f"Aspect Ratio:              {aspect_ratio if task_type == 't2v' else f'{width}:{height}'}\n"
        f"Video Length:              {video_length}\n"
        f"Reference Image:           {user_reference_image} {reference_image.size if reference_image is not None else ''}\n"
        f"Guidance Scale:            {guidance_scale}\n"
        f"Guidance Embedded Scale:   {embedded_guidance_scale}\n"
        f"Shift:                     {flow_shift}\n"
        f"Seed:                      {seed}\n"
        f"Video Resolution:          {width} x {height}\n"
        f"Attn mode:                 {pipe.transformer.attn_mode}\n"
        f"Transformer dtype:         {pipe.transformer.dtype}\n"
        f"Sampling Steps:            {num_inference_steps}\n"
        f"Use Meanflow:              {pipe.use_meanflow}\n"
        f"{'=' * 60}"
        "\n"
    )


def _encode_text_conditions(pipe, prompt, device, num_videos_per_prompt, negative_prompt):
    with auto_offload_model(pipe.text_encoder, pipe.execution_device, enabled=pipe.enable_offloading):
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
            clip_skip=pipe.clip_skip,
            data_type="video",
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

    with auto_offload_model(pipe.text_encoder_2, pipe.execution_device, enabled=pipe.enable_offloading):
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
            clip_skip=pipe.clip_skip,
            text_encoder=pipe.text_encoder_2,
            data_type="video",
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


def _prepare_byt5_kwargs(pipe, prompt, device):
    if not pipe.config.glyph_byT5_v2:
        return {}

    with auto_offload_model(pipe.byt5_model, pipe.execution_device, enabled=pipe.enable_offloading):
        return pipe._prepare_byt5_embeddings(prompt, device)


def _apply_classifier_free_guidance_embeddings(
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


def _prepare_latent_and_condition_inputs(
    pipe,
    batch_size,
    num_videos_per_prompt,
    latent_height,
    latent_width,
    latent_target_length,
    device,
    generator,
    task_type,
    reference_image,
    height,
    width,
    multitask_mask,
    semantic_images_np,
    target_resolution,
):
    num_channels_latents = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        latent_height,
        latent_width,
        latent_target_length,
        pipe.target_dtype,
        device,
        generator,
    )

    with auto_offload_model(pipe.vae, pipe.execution_device, enabled=pipe.enable_offloading):
        image_cond = pipe.get_image_condition_latents(task_type, reference_image, height, width)

    cond_latents = pipe._prepare_cond_latents(task_type, image_cond, latents, multitask_mask)
    with auto_offload_model(pipe.vision_encoder, pipe.execution_device, enabled=pipe.enable_offloading):
        vision_states = pipe._prepare_vision_states(
            semantic_images_np,
            target_resolution,
            latents,
            device,
        )

    return latents, cond_latents, vision_states


def _get_timestep_r(pipe, timesteps, step_index, latent_model_input):
    if not pipe.use_meanflow:
        return None

    next_timestep = (
        torch.tensor([0.0], device=pipe.execution_device)
        if step_index == len(timesteps) - 1
        else timesteps[step_index + 1]
    )
    return next_timestep.repeat(latent_model_input.shape[0])


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


def _run_denoising_loop(
    pipe,
    timesteps,
    num_inference_steps,
    latents,
    cond_latents,
    prompt_embeds,
    prompt_embeds_2,
    prompt_mask,
    vision_states,
    task_type,
    embedded_guidance_scale,
    predictor,
    step_list,
    extra_kwargs,
    extra_step_kwargs,
):
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)

    cache_helper = getattr(pipe, "cache_helper", None)
    if cache_helper is not None:
        cache_helper.clear_states()

    with pipe.progress_bar(total=num_inference_steps) as progress_bar, auto_offload_model(
        pipe.transformer,
        pipe.execution_device,
        enabled=pipe.enable_offloading,
    ):
        for i, t in enumerate(timesteps):
            if cache_helper is not None:
                cache_helper.cur_timestep = i

            latents_concat = torch.concat([latents, cond_latents], dim=1)
            latent_model_input = (
                torch.cat([latents_concat] * 2)
                if pipe.do_classifier_free_guidance
                else latents_concat
            )
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            t_expand = t.repeat(latent_model_input.shape[0])
            timesteps_r = _get_timestep_r(pipe, timesteps, i, latent_model_input)
            guidance_expand = _get_guidance_expand(
                embedded_guidance_scale,
                latent_model_input,
                pipe.execution_device,
                pipe.target_dtype,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=pipe.target_dtype,
                enabled=pipe.autocast_enabled,
            ):
                noise_pred = pipe.transformer(
                    latent_model_input,
                    t_expand,
                    prompt_embeds,
                    prompt_embeds_2,
                    prompt_mask,
                    timestep_r=timesteps_r,
                    vision_states=vision_states,
                    mask_type=task_type,
                    guidance=guidance_expand,
                    return_dict=False,
                    extra_kwargs=extra_kwargs,
                    cache_dic={"step": i, "step_list": step_list},
                    predictor=predictor,
                )[0]

            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                if pipe.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=pipe.guidance_rescale,
                    )

            latents = pipe.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0
            ):
                if progress_bar is not None:
                    progress_bar.update()

    return latents


def _run_super_resolution(
    pipe,
    prompt,
    sr_num_inference_steps,
    video_length,
    num_videos_per_prompt,
    seed,
    output_type,
    latents,
    user_reference_image,
    enable_vae_tile_parallelism,
):
    if not hasattr(pipe, "sr_pipeline"):
        raise AssertionError("sr_pipeline is required when enable_sr is True")

    return pipe.sr_pipeline(
        prompt=prompt,
        num_inference_steps=sr_num_inference_steps,
        video_length=video_length,
        negative_prompt="",
        num_videos_per_prompt=num_videos_per_prompt,
        seed=seed,
        output_type=output_type,
        lq_latents=latents,
        reference_image=user_reference_image,
        enable_vae_tile_parallelism=enable_vae_tile_parallelism,
    )


def _decode_video_frames(
    pipe,
    latents,
    output_type,
    generator,
    enable_sr,
    sr_out,
    return_pre_sr_video,
    enable_vae_tile_parallelism,
):
    if output_type == "latent":
        return latents

    if len(latents.shape) == 4:
        latents = latents.unsqueeze(2)
    elif len(latents.shape) != 5:
        raise ValueError(
            f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
        )

    if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor:
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        latents = latents / pipe.vae.config.scaling_factor

    if enable_vae_tile_parallelism and hasattr(pipe.vae, "enable_tile_parallelism"):
        pipe.vae.enable_tile_parallelism()

    if return_pre_sr_video or not enable_sr:
        with torch.autocast(
            device_type="cuda",
            dtype=pipe.vae_dtype,
            enabled=pipe.vae_autocast_enabled,
        ), auto_offload_model(
            pipe.vae,
            pipe.execution_device,
            enabled=pipe.enable_offloading,
        ), pipe.vae.memory_efficient_context():
            video_frames = pipe.vae.decode(latents, return_dict=False, generator=generator)[0]

        if video_frames is not None:
            video_frames = (video_frames / 2 + 0.5).clamp(0, 1).cpu().float()
        return video_frames

    return sr_out.videos


def _build_pipeline_output(video_frames, sr_out, enable_sr, return_dict):
    sr_video_frames = sr_out.videos if enable_sr else None

    if not return_dict:
        if enable_sr:
            return video_frames, sr_video_frames
        return video_frames

    if enable_sr:
        return HunyuanVideoPipelineOutput(videos=video_frames, sr_videos=sr_video_frames)
    return HunyuanVideoPipelineOutput(videos=video_frames)


def apply_disca(pipe):
    """Monkey-patch *transformer* with a cache-aware forward method.

    After calling this function the model exposes:
    - ``transformer.cache_img`` / ``transformer.cache_txt``: cached features.
    - The ``forward`` method accepts extra kwargs ``cache_dic`` and ``predictor``.
    """

    _initialize_transformer_cache(pipe.transformer)

    def new_cache_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        cache_dic: dict | None = None,
        predictor: nn.Module | None = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        guidance = _get_default_guidance(hidden_states, guidance)
        img, text_mask, t, tt, th, tw, freqs_cos, freqs_sin, parallel_dims = _prepare_transformer_inputs(
            self,
            hidden_states,
            timestep,
            encoder_attention_mask,
            freqs_cos,
            freqs_sin,
        )
        vec = _build_modulation_vector(self, t, text_states_2, guidance, timestep_r)
        txt = _embed_text_states(self, text_states, t, text_mask)
        txt, text_mask = _append_byt5_tokens(self, txt, text_mask, extra_kwargs)
        txt, text_mask = _append_vision_tokens(self, txt, text_mask, vision_states, mask_type)
        img, features_list = _compute_cache_aware_hidden_states(
            self,
            img,
            txt,
            vec,
            freqs_cos,
            freqs_sin,
            text_mask,
            cache_dic or {},
            predictor,
            output_features,
            output_features_stride,
        )
        img, features_list = _finalize_transformer_outputs(
            self,
            img,
            vec,
            parallel_dims,
            tt,
            th,
            tw,
            output_features,
            features_list,
        )
        assert return_dict is False, "return_dict is not supported."
        return (img, features_list)


    @torch.no_grad()
    def __call_disca__(
        self,
        prompt: Union[str, List[str]],
        aspect_ratio: str,
        video_length: int,
        prompt_rewrite: bool = True,
        num_inference_steps: int = None,
        guidance_scale: Optional[float] = None,
        enable_sr: bool = True,
        sr_num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        seed: Optional[int] = None,
        flow_shift: Optional[float] = None,
        embedded_guidance_scale: Optional[float] = None,
        reference_image=None,  # For i2v tasks: PIL Image or path to image file
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        return_pre_sr_video: bool = False,
        enable_vae_tile_parallelism: bool = True,
        predictor: nn.Module | None = None,
        step_list: List[int] = [0, 2, 4, 7],
        **kwargs,
    ):
        r"""
        Generates a video (or videos) based on text (and optionally image) conditions.

        Args:
            prompt (`str` or `List[str]`):
                Text prompt(s) to guide video generation.
            aspect_ratio (`str`):
                Output video aspect ratio as a string formatted like "720:1280" or "16:9". Required for text-to-video tasks.
            video_length (`int`):
                Number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps during generation. Larger values may improve video quality at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to value in config):
                Scale to encourage the model to better follow the prompt. `guidance_scale > 1` enables classifier-free guidance.
            enable_sr (`bool`, *optional*, defaults to True):
                Whether to apply super-resolution to the generated video.
            sr_num_inference_steps (`int`, *optional*, defaults to 30):
                Number of inference steps in the super-resolution module (if enabled).
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) that describe what should NOT be shown in the generated video.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                PyTorch random generator(s) for deterministic results.
            seed (`int`, *optional*):
                If specified, used to create the generator for reproducible sampling.
            flow_shift (`float`, *optional*):
                Flow shift parameter for the scheduler. Overrides the default pipeline configuration if provided.
            embedded_guidance_scale (`float`, *optional*):
                Additional control guidance scale, if supported.
            reference_image (PIL.Image or `str`, *optional*):
                Reference image for image-to-video (i2v) tasks. Can be a PIL image or a path to an image file. Set to `None` for text-to-video (t2v) generation.
            output_type (`str`, *optional*, defaults to "pt"):
                Output format of the returned video(s). Accepted values: `"pt"` for torch.Tensor or `"np"` for numpy.ndarray.
            return_dict (`bool`, *optional*, defaults to True):
                Whether to return a [`HunyuanVideoPipelineOutput`] or a tuple.
            **kwargs:
                Additional keyword arguments.

        Returns:
            HunyuanVideoPipelineOutput or `tuple`:
                If `return_dict` is True, returns a [`HunyuanVideoPipelineOutput`] with fields:
                    - `videos`: Generated video(s) as a tensor or numpy array.
                    - `sr_videos`: Super-resolved video(s) if `enable_sr` is True, else None.
                Otherwise, returns a tuple containing the outputs as above.

        Example:
            ```python
            pipe = HunyuanVideoPipeline.from_pretrained("your_model_dir")
            # Text-to-video
            video = pipe(prompt="A dog surfing on the beach", aspect_ratio="9:16", video_length=32).videos
            # Image-to-video
            video = pipe(prompt="Make this image move", reference_image="img.jpg", aspect_ratio="16:9", video_length=24).videos
            ```
        """
        num_videos_per_prompt = 1
        target_resolution = self.ideal_resolution
        guidance_scale, embedded_guidance_scale, flow_shift, num_inference_steps = (
            _resolve_generation_defaults(
                self,
                guidance_scale,
                embedded_guidance_scale,
                flow_shift,
                num_inference_steps,
            )
        )
        _validate_embedded_guidance(self, embedded_guidance_scale)

        user_reference_image = reference_image
        user_prompt = prompt
        task_type, reference_image, semantic_images_np = _prepare_task_inputs(reference_image)
        prompt = _maybe_rewrite_prompt(
            self,
            prompt_rewrite,
            user_prompt,
            reference_image,
            task_type,
        )

        if self.ideal_task is not None and self.ideal_task != task_type:
            raise ValueError(
                f"The loaded pipeline is trained for '{self.ideal_task}' task, but received input for '{task_type}' task. "
                "Please load a pipeline trained for the correct task, or check and update your arguments accordingly."
            )

        _configure_scheduler(self, flow_shift)
        seed, generator = _resolve_seed_and_generator(self, seed, generator)
        height, width = _resolve_generation_resolution(
            self,
            reference_image,
            aspect_ratio,
            target_resolution,
        )

        latent_target_length, latent_height, latent_width = self.get_latent_size(
            video_length,
            height,
            width,
        )
        n_tokens = latent_target_length * latent_height * latent_width
        multitask_mask = self.get_task_mask(task_type, latent_target_length)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        self._clip_skip = kwargs.get("clip_skip", None)

        batch_size = _get_batch_size(prompt)
        device = self.execution_device
        _log_generation_summary(
            self,
            user_prompt,
            prompt,
            prompt_rewrite,
            aspect_ratio,
            task_type,
            width,
            height,
            video_length,
            user_reference_image,
            reference_image,
            guidance_scale,
            embedded_guidance_scale,
            flow_shift,
            seed,
            num_inference_steps,
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = _encode_text_conditions(
            self,
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
        )
        extra_kwargs = _prepare_byt5_kwargs(self, prompt, device)
        prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2 = (
            _apply_classifier_free_guidance_embeddings(
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

        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps,
            {"n_tokens": n_tokens},
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **extra_set_timesteps_kwargs,
        )
        latents, cond_latents, vision_states = _prepare_latent_and_condition_inputs(
            self,
            batch_size,
            num_videos_per_prompt,
            latent_height,
            latent_width,
            latent_target_length,
            device,
            generator,
            task_type,
            reference_image,
            height,
            width,
            multitask_mask,
            semantic_images_np,
            target_resolution,
        )
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": kwargs.get("eta", 0.0)},
        )
        latents = _run_denoising_loop(
            self,
            timesteps,
            num_inference_steps,
            latents,
            cond_latents,
            prompt_embeds,
            prompt_embeds_2,
            prompt_mask,
            vision_states,
            task_type,
            embedded_guidance_scale,
            predictor,
            step_list,
            extra_kwargs,
            extra_step_kwargs,
        )

        sr_out = None
        if enable_sr:
            sr_out = _run_super_resolution(
                self,
                prompt,
                sr_num_inference_steps,
                video_length,
                num_videos_per_prompt,
                seed,
                output_type,
                latents,
                user_reference_image,
                enable_vae_tile_parallelism,
            )

        video_frames = _decode_video_frames(
            self,
            latents,
            output_type,
            generator,
            enable_sr,
            sr_out,
            return_pre_sr_video,
            enable_vae_tile_parallelism,
        )
        self.maybe_free_model_hooks()
        return _build_pipeline_output(video_frames, sr_out, enable_sr, return_dict)

    pipe.__class__.__call__ = __call_disca__

    pipe.transformer.forward = MethodType(new_cache_forward, pipe.transformer)
