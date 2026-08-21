# Embedding model fine-tuning pipeline

## For Embedding model fine-tuning

### Dataset

Any Structured Datasets(default setting is triplet)

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/embedding-fine-tune-hf.git
cd embedding-fine-tune-hf

# [OPTIONAL] create conda environment
conda create -n myenv python=3.12 -y
conda activate myenv

# install requirements with the validated CUDA backend
python -m pip install uv==0.10.12
uv pip install \
    --torch-backend=cu129 \
    -r requirements.txt
```

### .env file setting

```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}
HF_HOME={HF_HOME}
USER_NAME={USER_NAME}
```

### Train

* end-to-end

```shell
python main.py mode=train
```

### Examples of shell scipts

* train

```shell
bash scripts/train/train.sh
```

### Additional Options

* LoRA PEFT option

```shell
is_peft={True or False}
```

* Upload user name and model name at HuggingFace Model card

```shell
upload_user={upload_user} 
model_type={model_type}
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__

### Quick setup (pyproject.toml)

```bash
# install project dependencies from pyproject.toml
python -m pip install uv==0.10.12
uv pip install --torch-backend=cu129 .

# [OPTIONAL] editable install for development
uv pip install --torch-backend=cu129 -e .
```
