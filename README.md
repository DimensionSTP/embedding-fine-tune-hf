# LLM model scaling pipeline

## For (s)LLM model scaling

### Dataset

Any Structured Datasets

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/embedding-fine-tune-hf.git
cd embedding-fine-tune-hf

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# install requirements
pip install -r requirements.txt
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

### Test

* end-to-end

```shell
python main.py mode=test
```

* end-to-end(vLLM)

```shell
python main.py mode=test_vllm
```

### Examples of shell scipts

* full preprocessing

```shell
bash scripts/preprocess.sh
```

* dataset preprocessing

```shell
bash scripts/preprocess_dataset.sh
```

* train

```shell
bash scripts/train.sh
```

* test

```shell
bash scripts/test.sh
```

* test_vllm

```shell
bash scripts/test_vllm.sh
```

### Additional Options

* Pure decoder based LLM LoRA or QLoRA PEFT option

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
