#export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
#export T2V_REWRITE_MODEL_NAME="<your_model_name>"
#export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
#export I2V_REWRITE_MODEL_NAME="<your_model_name>"
# uncomment to use your rewrite model

PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

IMAGE_PATH=example.JPG # Optional, none or <image path> to enable i2v mode
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p
OUTPUT_PATH=./outputs/output.mp4
MODEL_PATH=/path/to/HunyuanVideo-1.5/ckpts # Path to pretrained model
#CHECKPOINT_PATH=/path/to/your/hyvideo15_i2v_dit_480p.safetensors
PREDICTOR_PATH=/path/to/hyvideo15_i2v_predictor_480p.safetensors # Path to predictor checkpoint

STEP_LIST="0 2 4 7" # full step list for DisCa, tune this for your task.

# Configuration for faster inference
N_INFERENCE_GPU=8 # Parallel inference GPU count
CFG_DISTILLED=true # Inference with CFG distilled model, 2x speedup
SAGE_ATTN=true # Inference with SageAttention
SPARSE_ATTN=false # Inference with sparse attention (only 720p models are equipped with sparse attention). Please ensure flex-block-attn is installed
OVERLAP_GROUP_OFFLOADING=true # Only valid when group offloading is enabled, significantly increases CPU memory usage but speeds up inference
ENABLE_CACHE=false # Enable feature cache during inference. Significantly speeds up inference.
CACHE_TYPE=taylorcache # Support: deepcache, teacache, taylorcache
ENABLE_STEP_DISTILL=true # Enable step distilled model for 480p I2V, recommended 8 or 12 steps, up to 6x speedup
NUM_INFERENCE_STEPS=8 # Total inference steps, recommended 8 or 12 for step distilled model


# Configuration for better quality
REWRITE=true # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution


torchrun --nproc_per_node=$N_INFERENCE_GPU generate_disca.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --rewrite $REWRITE \
  --cfg_distilled $CFG_DISTILLED \
  --enable_step_distill $ENABLE_STEP_DISTILL \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --sr $ENABLE_SR --save_pre_sr_video \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --predictor_path $PREDICTOR_PATH \
  --step_list $STEP_LIST \
  --num_inference_steps $NUM_INFERENCE_STEPS