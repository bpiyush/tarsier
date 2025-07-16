Follow these steps to run video captioning.

1. Install dependencies

```bash
conda create -n tarsier python=3.9 -y
conda activate tarsier


git clone https://github.com/bpiyush/tarsier.git && cd tarsier
git checkout tarsier2

pip install -r requirements.txt
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install pyarrow
pip install 'accelerate>=0.26.0'
```

2. Download the model.

```sh
git lfs install
git clone https://huggingface.co/omni-research/Tarsier2-Recap-7b

# Check example file
cat preprocessor_config.json

# if this file shows lfs configuration instead of a dict config, then run
git lfs --version
git lfs install
git lfs pull
```

3. Run the script

```sh
VIDEO_FILE="assets/videos/coffee.gif"
MODEL_NAME_OR_PATH="/work/piyush/pretrained_checkpoints/Tarsier2/Tarsier2-Recap-7b"
python3 -m tasks.inference_quick_start   \
--model_name_or_path $MODEL_NAME_OR_PATH   \
--config configs/tarser2_default_config.yaml   \
--instruction "Describe the video in detail."   \
--input_path $VIDEO_FILE

# A hand picks up a cup of coffee with a heart-shaped design on the foam from a table surrounded by red roses. The person lifts the cup and takes a sip while holding a book in the other hand. The person then turns their head to the left, looking towards two other individuals sitting at a table in the background. The two individuals in the background wave a nd make gestures towards the person with the coffee. The scene gradually fades out, leaving the background with the Eiffel Tower and roses.
```
