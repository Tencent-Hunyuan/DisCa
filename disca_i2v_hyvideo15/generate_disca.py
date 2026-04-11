# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import copy
import datetime
import json

import loguru
import torch
import argparse
import einops
import imageio
from torch import distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.disca_utils import apply_disca, Predictor
from safetensors.torch import load_file as safetensors_load_file

parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=24)

def rank0_log(message, level):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.log(level, message)

def save_config(args, output_path, task, transformer_version):
    arguments = {}
    for key, value in vars(args).items():
        if not key.startswith('_') and not callable(value):
            try:
                json.dumps(value)
                arguments[key] = value
            except (TypeError, ValueError):
                arguments[key] = str(value)
    
    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task': task,
        'transformer_version': transformer_version,
        'output_path': output_path,
        'arguments': arguments
    }
    
    base_path, _ = os.path.splitext(output_path)
    config_path = f"{base_path}_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved generation config to: {config_path}")
    return config_path

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

def load_checkpoint_to_transformer(pipe, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    rank0_log(f"Loading checkpoint from {checkpoint_path}", "INFO")
    
    try:
        model_state_dict = get_model_state_dict(pipe.transformer)
        dcp.load(
            state_dict={"model": model_state_dict},
            checkpoint_id=checkpoint_path,
        )
        rank0_log("Transformer model state loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading checkpoint: {e}", "ERROR")
        raise

def load_lora_adapter(pipe, lora_path):
    rank0_log(f"Loading LoRA adapter from {lora_path}", "INFO")
    try:
        pipe.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_path,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
        rank0_log("LoRA adapter loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading LoRA adapter: {e}", "ERROR")
        raise
    

PREDICTOR_HIDDEN_SIZE = 2048
PREDICTOR_HEADS = 16
PREDICTOR_DEPTH = 2
PREDICTOR_MLP_RATIO = 4.0


def load_predictor(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Predictor:
    """Build and load a Predictor from a safetensors checkpoint."""
    predictor = Predictor(
        hidden_size=PREDICTOR_HIDDEN_SIZE,
        heads_num=PREDICTOR_HEADS,
        model_depth=PREDICTOR_DEPTH,
        mlp_width_ratio=PREDICTOR_MLP_RATIO,
    )

    if checkpoint_path.endswith(".safetensors"):
        state_dict = safetensors_load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    predictor.load_state_dict(state_dict, strict=False)
    predictor = predictor.to(device=device, dtype=dtype)
    predictor.eval()
    predictor.requires_grad_(False)
    return predictor


def _validate_generate_args(args):
    if args.sparse_attn and args.use_sageattn:
        raise ValueError(
            "sparse_attn and use_sageattn cannot be enabled simultaneously. Please enable only one of them."
        )

    if args.use_fp8_gemm and "sgl" in args.quant_type:
        try:
            import sgl_kernel  # noqa: F401
        except Exception:
            raise ValueError(
                "sgl_kernel is not installed. Please install it using `pip install sgl-kernel==0.3.18`"
            )

    if args.enable_step_distill and args.enable_cache:
        raise ValueError(
            "Enabling both step distilled model and cache will lead to performance degradation."
        )


def _resolve_task(task_image_path):
    return "i2v" if task_image_path else "t2v"


def _resolve_transformer_dtype(dtype_name):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}. Must be 'bf16' or 'fp32'")


def _resolve_offloading_settings(args):
    enable_offloading = args.offloading
    if args.group_offloading is None:
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config["enable_group_offloading"]
    else:
        enable_group_offloading = args.group_offloading

    overlap_group_offloading = args.overlap_group_offloading
    device = torch.device("cpu") if enable_offloading else torch.device("cuda")
    transformer_init_device = (
        torch.device("cpu") if enable_group_offloading else device
    )

    return {
        "enable_offloading": enable_offloading,
        "enable_group_offloading": enable_group_offloading,
        "overlap_group_offloading": overlap_group_offloading,
        "device": device,
        "transformer_init_device": transformer_init_device,
    }


def _create_and_prepare_pipeline(
    args,
    infer_state,
    transformer_version,
    transformer_dtype,
    enable_sr,
    offloading_settings,
):
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=enable_sr,
        transformer_dtype=transformer_dtype,
        device=offloading_settings["device"],
        transformer_init_device=offloading_settings["transformer_init_device"],
    )

    loguru.logger.info(
        f"{offloading_settings['enable_offloading']=} "
        f"{offloading_settings['enable_group_offloading']=} "
        f"{offloading_settings['overlap_group_offloading']=}"
    )

    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=offloading_settings["enable_offloading"],
        enable_group_offloading=offloading_settings["enable_group_offloading"],
        overlap_group_offloading=offloading_settings["overlap_group_offloading"],
    )
    apply_disca(pipe)

    if args.checkpoint_path:
        load_checkpoint_to_transformer(pipe, args.checkpoint_path)
    if args.lora_path:
        load_lora_adapter(pipe, args.lora_path)

    return pipe


def _load_required_predictor(args, device, transformer_dtype):
    if not args.predictor_path:
        raise ValueError(
            "Predictor path is required. Please provide a predictor path using --predictor_path"
        )
    return load_predictor(args.predictor_path, device=device, dtype=transformer_dtype)


def _maybe_prepare_sr_pipeline(pipe, infer_state, enable_sr, offloading_settings):
    if not (enable_sr and hasattr(pipe, "sr_pipeline")):
        return

    sr_infer_state = copy.deepcopy(infer_state)
    sr_infer_state.enable_cache = False
    pipe.sr_pipeline.apply_infer_optimization(
        infer_state=sr_infer_state,
        enable_offloading=offloading_settings["enable_offloading"],
        enable_group_offloading=offloading_settings["enable_group_offloading"],
        overlap_group_offloading=offloading_settings["overlap_group_offloading"],
    )


def _build_generation_extra_kwargs(task, image_path):
    extra_kwargs = {}
    if task == "i2v":
        extra_kwargs["reference_image"] = image_path
    return extra_kwargs


def _warn_generation_settings(args):
    if args.video_length != 121:
        rank0_log(
            f"Warning: 121 frames is the optimal value for best quality. "
            f"Attempting to generate {args.video_length} frames...",
            "WARNING",
        )

    if not args.rewrite:
        rank0_log(
            "Warning: Prompt rewriting is disabled. This may affect the quality of generated videos.",
            "WARNING",
        )


def _run_generation(pipe, args, enable_sr, predictor, extra_kwargs):
    return pipe(
        enable_sr=enable_sr,
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.num_inference_steps,
        sr_num_inference_steps=None,
        video_length=args.video_length,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=args.rewrite,
        return_pre_sr_video=args.save_pre_sr_video,
        predictor=predictor,
        step_list=args.step_list,
        **extra_kwargs,
    )


def _resolve_output_path(args, transformer_version):
    output_path = args.output_path
    if output_path is None:
        now = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
        output_path = f"./outputs/output_{transformer_version}_{now}.mp4"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_path


def _save_generation_outputs(args, out, enable_sr, output_path):
    if enable_sr and hasattr(out, "sr_videos"):
        save_video(out.sr_videos, output_path)
        print(f"Saved SR video to: {output_path}")

        if args.save_pre_sr_video:
            base_path, ext = os.path.splitext(output_path)
            original_path = f"{base_path}_before_sr{ext}"
            save_video(out.videos, original_path)
            print(f"Saved original video (before SR) to: {original_path}")
        return

    save_video(out.videos, output_path)
    print(f"Saved video to: {output_path}")


def _maybe_save_generation_config(args, output_path, task, transformer_version):
    if not args.save_generation_config:
        return

    try:
        save_config(args, output_path, task, transformer_version)
    except Exception:
        pass


def generate_video(args):
    infer_state = initialize_infer_state(args)
    _validate_generate_args(args)

    task = _resolve_task(args.image_path)
    enable_sr = args.sr
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution,
        task,
        args.cfg_distilled,
        args.enable_step_distill,
        args.sparse_attn,
    )
    transformer_dtype = _resolve_transformer_dtype(args.dtype)
    offloading_settings = _resolve_offloading_settings(args)
    pipe = _create_and_prepare_pipeline(
        args,
        infer_state,
        transformer_version,
        transformer_dtype,
        enable_sr,
        offloading_settings,
    )
    predictor = _load_required_predictor(
        args,
        device=offloading_settings["device"],
        transformer_dtype=transformer_dtype,
    )
    _maybe_prepare_sr_pipeline(
        pipe,
        infer_state,
        enable_sr,
        offloading_settings,
    )

    extra_kwargs = _build_generation_extra_kwargs(task, args.image_path)
    _warn_generation_settings(args)
    out = _run_generation(pipe, args, enable_sr, predictor, extra_kwargs)

    if int(os.environ.get('RANK', '0')) == 0:
        output_path = _resolve_output_path(args, transformer_version)
        _save_generation_outputs(args, out, enable_sr, output_path)
        _maybe_save_generation_config(args, output_path, task, transformer_version)

def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanVideo-1.5')

    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for video generation (default: empty string)'
    )
    parser.add_argument(
        '--resolution', type=str, required=True, choices=['480p', '720p'],
        help='Video resolution (480p or 720p)'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--aspect_ratio', type=str, default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=None,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--video_length', type=int, default=121,
        help='Number of frames to generate (default: 121)'
    )
    parser.add_argument(
        '--sr', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable super resolution (default: true). '
             'Use --sr or --sr true/1 to enable, --sr false/0 to disable'
    )
    parser.add_argument(
        '--save_pre_sr_video', type=str_to_bool, nargs='?', const=True, default=False,
        help='Save original video before super resolution (default: false). '
             'Use --save_pre_sr_video or --save_pre_sr_video true/1 to enable, '
             '--save_pre_sr_video false/0 to disable'
    )
    parser.add_argument(
        '--rewrite', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable prompt rewriting (default: true). '
             'Use --rewrite or --rewrite true/1 to enable, --rewrite false/0 to disable'
    )
    parser.add_argument(
        '--cfg_distilled', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable CFG distilled model (default: false). '
             'Use --cfg_distilled or --cfg_distilled true/1 to enable, '
             '--cfg_distilled false/0 to disable'
    )
    parser.add_argument(
        '--enable_step_distill', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable step distilled model (default: false). '
             'Use --enable_step_distill or --enable_step_distill true/1 to enable, '
             '--enable_step_distill false/0 to disable'
    )
    parser.add_argument(
        '--sparse_attn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sparse attention (default: false). '
             'Use --sparse_attn or --sparse_attn true/1 to enable, '
             '--sparse_attn false/0 to disable'
    )
    parser.add_argument(
        '--offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable offloading (default: true). '
             'Use --offloading or --offloading true/1 to enable, '
             '--offloading false/0 to disable'
    )
    parser.add_argument(
        '--group_offloading', type=str_to_bool, nargs='?', const=True, default=None,
        help='Enable group offloading (default: None, automatically enabled if offloading is enabled). '
             'Use --group_offloading or --group_offloading true/1 to enable, '
             '--group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--overlap_group_offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable overlap group offloading (default: true). '
             'Significantly increases CPU memory usage but speeds up inference. '
             'Use --overlap_group_offloading or --overlap_group_offloading true/1 to enable, '
             '--overlap_group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for transformer (default: bf16). '
             'bf16: faster, lower memory; fp32: better quality, slower, higher memory'
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Random seed (default: 123)'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided, uses i2v mode)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Output file path for generated video (if not provided, saves to ./outputs/output.mp4)'
    )
    parser.add_argument(
        '--use_sageattn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sageattn (default: false). '
             'Use --use_sageattn or --use_sageattn true/1 to enable, '
             '--use_sageattn false/0 to disable'
    )
    parser.add_argument(
        '--sage_blocks_range', type=str, default="0-53",
        help='Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--enable_torch_compile', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable torch compile for transformer (default: false). '
             'Use --enable_torch_compile or --enable_torch_compile true/1 to enable, '
             '--enable_torch_compile false/0 to disable'
    )
    parser.add_argument(
        '--enable_cache', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable cache for transformer (default: false). '
             'Use --enable_cache or --enable_cache true/1 to enable, '
             '--enable_cache false/0 to disable'
    )
    parser.add_argument(
        '--cache_type', type=str, default="deepcache",
        help='Cache type for transformer (e.g., deepcache, teacache, taylorcache)'
    )
    parser.add_argument(
        '--no_cache_block_id', type=str, default="53",
        help='Blocks to exclude from deepcache (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--cache_start_step', type=int, default=11,
        help='Start step to skip when using cache (default: 11)'
    )
    parser.add_argument(
        '--cache_end_step', type=int, default=45,
        help='End step to skip when using cache (default: 45)'
    )
    parser.add_argument(
        '--total_steps', type=int, default=50,
        help='Total inference steps (default: 50)'
    )
    parser.add_argument(
        '--cache_step_interval', type=int, default=4,
        help='Step interval to skip when using cache (default: 4)'
    )
    parser.add_argument(
        '--save_generation_config', type=str_to_bool, nargs='?', const=True, default=True,
        help='Save generation config file (default: true). '
             'Use --save_generation_config or --save_generation_config true/1 to enable, '
             '--save_generation_config false/0 to disable'
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to checkpoint directory containing transformer weights (e.g., ./outputs/checkpoint-1000/transformer). '
             'The checkpoint directory should contain a "transformer" subdirectory. '
             'If provided, the transformer model weights will be loaded from this checkpoint.'
    )
    parser.add_argument(
        '--lora_path', type=str, default=None,
        help='Path to LoRA adapter directory or checkpoint directory containing LoRA adapter. '
             'If provided, the LoRA adapter will be loaded to the transformer model.'
    )

    # fp8 gemm related
    parser.add_argument(
        '--use_fp8_gemm', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable fp8 gemm for transformer (default: false). '
             'Use --use_fp8_gemm or --use_fp8_gemm true/1 to enable, '
             '--use_fp8_gemm false/0 to disable'
    )
    parser.add_argument(
        '--quant_type', type=str, default="fp8-per-token-sgl",
        help='Quantization type for fp8 gemm (e.g., fp8-per-tensor-weight-only, fp8-per-tensor, fp8-per-token-sgl)'
    )
    parser.add_argument(
        '--include_patterns', type=str, default="double_blocks",
        help='Include patterns for fp8 gemm (default: double_blocks)'
    )

    # DisCa Config
    parser.add_argument(
        '--predictor_path', type=str, default=None,
        help='Path to predictor checkpoint (.safetensors or .pt)'
    )
    parser.add_argument(
        '--step_list', type=int, nargs='+', default=[0, 2, 4, 7],
        help='Step list for DisCa (default: [0, 2, 4, 7])'
    )

    args = parser.parse_args()
    
    # Convert string "none" to None for image_path
    if args.image_path is not None and args.image_path.lower().strip() == 'none':
        args.image_path = None
    
    
    generate_video(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
