export MODEL_BASE=/ckpts/for/HunyuanVideo

python3 sample_video.py \
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
	--save-path ./results
