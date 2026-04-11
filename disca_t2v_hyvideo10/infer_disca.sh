export MODEL_BASE=/ckpts/for/HunyuanVideo

python3 sample_video_disca.py \
	--model HYVideo-T/2-meanflow \
	--model-base /ckpts/for/HunyuanVideo \
	--dit-weight /path/to/DisCa/hyvideo10_t2v/hyvideo10_t2v_restricted_meanflow_540p.safetensors \
	--predictor-path /path/to/DisCa/hyvideo10_t2v/hyvideo10_t2v_predictor_540p.safetensors \
	--video-size 704 704 \
	--video-length 129 \
	--infer-steps 20 \
	--step-list 0 1 2 5 9 11 13 15 17 19 \
	--prompt "A cat." \
	--seed 42 \
	--embedded-cfg-scale 6.0 \
	--flow-shift 5.0 \
	--flow-reverse \
	--save-path ./results

# example step: 
# 10 step:0 1 2 5 9 11 13 15 17 19
# 8 steps: 0 1 2 5 9 13 17 19
# just tune for your task. 
# This repo just propose a potential direction of combining distillation with cache. 
# Better training strategy for predictor will gain better performance.
# Regretfully, i do not have that much time. TAT.