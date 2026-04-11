# DisCa for HunyuanVideo1.5

## 🛠️ Dependencies and Installation

### Step 1: Clone HunyuanVideo1.5

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
cd HunyuanVideo-1.5
```

### Step 2: Install Basic Dependencies

```bash
pip install -r requirements.txt
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
```

### Step 3: Install Attention Libraries

* Flash Attention: 
  Install Flash Attention for faster inference and reduced GPU memory consumption.
  Detailed installation instructions are available at [Flash Attention](https://github.com/Dao-AILab/flash-attention).

* Flex-Block-Attention: 
  flex-block-attn is only required for sparse attention to achieve faster inference and can be installed by the following command:
  ```bash
  git clone https://github.com/Tencent-Hunyuan/flex-block-attn.git
  cd flex-block-attn
  git submodule update --init --recursive
  python3 setup.py install
  ```

* SageAttention: 
  To enable SageAttention for faster inference, you need to install it by the following command:
  > **Note**: Enabling SageAttention will automatically disable Flex-Block-Attention.
  ```bash
  git clone https://github.com/cooper1637/SageAttention.git
  cd SageAttention 
  export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
  python3 setup.py install
  ```

### ✨Step 4: Install DisCa for HunyuanVideo1.5
* Copy disca_i2v_hyvideo15/disca_utils -> HunyuanVideo1.5/hyvideo/disca_utils.

    ```bash
    cp -r DisCa/disca_i2v_hyvideo15/disca_utils HunyuanVideo1.5/hyvideo/disca_utils
    ```
* Copy generation scripts.
    ```bash
    cp disca_i2v_hyvideo15/generate_disca.py HunyuanVideo1.5/generate_disca.py
    cp disca_i2v_hyvideo15/infer_disca.sh HunyuanVideo1.5/infer_disca.sh
    ```

## 🧱 Download Pretrained Models

Download the pretrained models before generating videos. Detailed instructions are available at [checkpoints-download.md](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/checkpoints-download.md).

|ModelName| Download                     |
|-|---------------------------| 
|HunyuanVideo-1.5-480P-I2V-step-distill |[480P-I2V-step-distill](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v_step_distilled) |
|HunyuanVideo-1.5-480P-I2V-step-distill-DisCa |[480P-I2V-step-distill-DisCa](placeholder)|

## 🚀 Inference with Source Code

### Multi-Device Inference (fast)
```bash
cd HunyuanVideo-1.5
bash infer_disca.sh
'''
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
'''
```