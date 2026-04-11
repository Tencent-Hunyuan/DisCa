export TOKENIZERS_PARALLELISM=false

export NPROC_PER_NODE=4
export ULYSSES_DEGREE=4
export RING_DEGREE=1
export MODEL_BASE=/path/to/HunyuanVideo/models

torchrun --nproc_per_node=$NPROC_PER_NODE sample_video.py \
	--model HYVideo-T/2-meanflow \
	--model-base /ckpts/for/HunyuanVideo \
	--dit-weight /path/to/DisCa/hyvideo10_t2v/hyvideo10_t2v_restricted_meanflow_540p.safetensors \
	--video-size 704 704 \
	--video-length 129 \
	--infer-steps 20 \
	--prompt "A cat." \
	--seed 42 \
	--embedded-cfg-scale 6.0 \
	--flow-shift 5.0 \
	--flow-reverse \
	--ulysses-degree=$ULYSSES_DEGREE \
	--ring-degree=$RING_DEGREE \
	--save-path ./results
