import os
import time
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from hyvideo.config import parse_args
from hyvideo.disca_utils.apply_disca import apply_disca
from hyvideo.disca_utils.predictor import Predictor
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import save_videos_grid


def _unwrap_predictor_state_dict(state_dict):
    for key in ("model", "module", "predictor"):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]

    if state_dict and all(key.startswith("predictor.") for key in state_dict):
        state_dict = {key[len("predictor."):]: value for key, value in state_dict.items()}
    elif state_dict and all(key.startswith("module.predictor.") for key in state_dict):
        state_dict = {
            key[len("module.predictor."):]: value for key, value in state_dict.items()
        }
    elif state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    return state_dict


def load_predictor(checkpoint_path, device, dtype):
    if checkpoint_path is None:
        raise ValueError("`--predictor-path` is required for DisCa inference.")

    predictor_path = Path(checkpoint_path)
    if not predictor_path.exists():
        raise ValueError(f"Predictor checkpoint not found: {predictor_path}")

    predictor = Predictor()

    if predictor_path.suffix == ".safetensors":
        from safetensors.torch import load_file as safetensors_load_file

        logger.info(f"Loading predictor safetensors from: {predictor_path}")
        state_dict = safetensors_load_file(str(predictor_path), device="cpu")
    else:
        logger.info(f"Loading predictor torch checkpoint from: {predictor_path}")
        state_dict = torch.load(predictor_path, map_location="cpu")

    state_dict = _unwrap_predictor_state_dict(state_dict)
    state_dict, is_converted = HunyuanVideoSampler._convert_tensor_parallel_state_dict_to_hunyuan(state_dict)
    if is_converted:
        logger.info("Converted tensor-parallel predictor weights to the merged Hunyuan format.")

    predictor.load_state_dict(state_dict, strict=True)
    predictor = predictor.to(device=device, dtype=dtype)
    predictor.eval()
    predictor.requires_grad_(False)
    return predictor


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    save_path = (
        args.save_path
        if args.save_path_suffix == ""
        else f"{args.save_path}_{args.save_path_suffix}"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    apply_disca(hunyuan_video_sampler)

    predictor = load_predictor(
        checkpoint_path=args.predictor_path,
        device=hunyuan_video_sampler.device,
        dtype=hunyuan_video_sampler.model.dtype,
    )

    outputs = hunyuan_video_sampler.predict_disca(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        predictor=predictor,
        step_list=args.step_list,
    )
    samples = outputs["samples"]

    if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            prompt_name = outputs["prompts"][i][:100].replace("/", "")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{prompt_name}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f"Sample save to: {cur_save_path}")


if __name__ == "__main__":
    main()
