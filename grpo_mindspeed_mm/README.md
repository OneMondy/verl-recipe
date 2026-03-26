# Reinforcement learning for qwen3.5 models using MindSpeed-mm as the backend
<p align="center">

## 1. Environment installation ##

\[You are advised to use the matching environment version during model development.\]

For details, see [Installation Guide](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md).

```shell
# Importing CANN Environment Variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Creating the python3.11 conda environment
conda create -n test python=3.11
conda activate test

# Install vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout a75a5b54c7f76bc2e15d3025d6
git fetch origin pull/34521/head:pr-34521
git merge pr-34521
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout c63b7a11888e9e1caeeff8
git fetch origin pull/6742/head:pr-6742
git merge pr-6742
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e .
cd ..

# Install Verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 3bbdecee9a243388442b800326d57a4f3dc41516
pip install -e .
cd ..

# Update the recipe directory.
git clone https://github.com/verl-project/verl-recipe.git
mkdir verl/recipe/grpo_mindspeed_mm
cp -rf verl-recipe/grpo_mindspeed_mm verl/recipe/

# Installing the Mindspeed-MM
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
git checkout 7b8e497efcbfb1a44a77a03a7755de88f1a0424a
git cherry-pick b8cbe78745ab1fe2f7ef02292a0c3c50a5174ee5 (pr_2292)
bash scripts/install.sh --msid eb10b92 && bash examples/fsdp2/qwen3_5/install_extensions.sh
# torch version mismatch detected. Reinstall PyTorch? (y/n) -> n
# Reinstall torch_npu to match PyTorch version? (y/n) -> n
cd ..

# Installing Other Packages
pip install torch_npu==2.9.0 torchvision==0.24.0 mathruler


# The directory structure after the preparation is as follows:
# MindSpeed-MM
# verl-recipe
# verl
# ├── recipe
#     ├── grpo_mindspeed_mm
# vllm
# vllm-ascend
```

## 2. Dataset preparation ##

recommend use geo3k dataset. 


## 3. Training model preparation ##

Qwen3.5 27B model download address:

https://huggingface.co/Qwen/Qwen3.5-27B

The downloaded model is in the huggingface format and needs to be converted to the dcp format for training. For details, see the following section. 

### convert HF weight to DCP weight ###

1.  Weight of the downloaded Qwen3.5 model In the mm root directory, run the following script to convert the weight:

```shell
cd MindSpeed-MM

mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ckpt/hf_path/xxxxxxx \
--dcp_dir ckpt/dcp_path/xxxxxxx

# 转换后的目录结构为：
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```


Parameters in the weight conversion script are described as follows:

| Parameters        | Meaning:                                                  |
| ----------------- | --------------------------------------------------------- |
| --hf_dir | Original weight path of the huggingface                                      |
| --dcp_dir | Path for storing weights after conversion or segmentation |


## 5. Parameters for configuring args ##

Modify the following parameters and run the script to generate the args file for training preparation:

| Configuration File                                                   | Modifying a field  | Modification Description                                                  |
|----------------------------------------------------------------------|--------------------|---------------------------------------------------------------------------|
| verl/recipe/grpo_mindspeed_mm/examples/qwen3_5_27B_config.yaml | model_name_or_path | Huggingface weight path                                                   |
| verl/recipe/grpo_mindspeed_mm/examples/qwen3_5_27B_config.yaml | load               | DCP weight path                                                           |
| verl/recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh    | MODEL_PATH          | Huggingface weight path |
| verl/recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh       | TRAIN_FILE      | dataset for train                                                         |
| verl/recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh       | TEST_FILE      | dataset for test                                                          |


```shell
# source /usr/local/Ascend/cann/set_env.sh
# cd verl
bash recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh
```
