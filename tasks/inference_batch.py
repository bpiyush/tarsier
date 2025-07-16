"""Inference script to caption a batch of videos."""
from tasks.utils import load_model_and_processor
from dataset.custom_data_parsers.utils import (
    put_pred_to_data_dict,
    get_prompt_from_data_dict,
)
from dataset.utils import *

import os
import torch
from tqdm import tqdm
import yaml


def process_single_video(model, processor, prompt, video_file, generate_kwargs):
    """
    Processes a single video and returns the caption.
    """
    sample = format_one_sample(video_file, prompt)
    batch_data = processor(sample)

    print(f"###Prompt:\n{get_prompt_from_data_dict(sample)}")
    model_inputs = {}
    for k, v in batch_data.items():
        if not isinstance(v, torch.Tensor):
            continue
        model_inputs[k] = v.to(model.device)
    outputs = model.generate(
        **model_inputs,
        **generate_kwargs,
    )
    output_text = processor.processor.tokenizer.decode(
        outputs[0][model_inputs['input_ids'][0].shape[0]:],
        skip_special_tokens=True,
    )
    return output_text


def process_multiple_videos(model, processor, prompt, video_files, generate_kwargs):
    """
    Processes multiple videos and returns the captions.
    Since the prompt is the same for all videos, we can process them in a batch.
    
    Args:
        video_files: list of video files to process
    """
    
    # Format each video one by one
    samples = [format_one_sample(video_file, prompt) for video_file in video_files]
    
    # Create a batch of samples
    batch_data = processor(samples)
    model_inputs = {}
    for k, v in batch_data.items():
        if not isinstance(v, torch.Tensor):
            continue
        model_inputs[k] = v.to(model.device)

    # Forward pass the entire batch
    outputs = model.generate(
        **model_inputs,
        **generate_kwargs,
    )

    # Decode the outputs one by one
    captions = []
    for i in range(len(video_files)):
        caption = processor.processor.tokenizer.decode(
            outputs[i][model_inputs['input_ids'][i].shape[0]:],
            skip_special_tokens=True,
        )
        captions.append(caption)

    return captions


if __name__ == "__main__":
    
    # Hyperparameters and configs
    config = "configs/tarser2_default_config.yaml"
    model_name_or_path = "/work/piyush/pretrained_checkpoints/Tarsier2/Tarsier2-Recap-7b/"
    temperature = 0.
    generate_kwargs = {
        "do_sample": True if temperature > 0 else False,
        "max_new_tokens": 256,
        "top_p": 1,
        "temperature": temperature,
        "use_cache": True
    }
    prompt = "Describe the video in detail."
    
    # Load model and processor
    data_config = yaml.safe_load(open(config, 'r'))
    model, processor = load_model_and_processor(
        model_name_or_path, data_config=data_config,
    )


    # Debugging on a single video first
    debug = True
    if debug:
        # video_file = "assets/videos/coffee.gif"
        video_file = "/scratch/shared/beegfs/piyush/datasets/EPIC-Kitchens-100/P01_105_181.3_184.7.mp4"
        caption = process_single_video(
            model, processor, prompt, video_file, generate_kwargs,
        )
        import ipdb; ipdb.set_trace()

        # Test on a batch of videos
        video_files = [
            "assets/videos/coffee.gif",
            "assets/videos/demo_test.mp4",
            "assets/videos/sitting.mp4",
        ]
        captions = process_multiple_videos(
            model, processor, prompt, video_files, generate_kwargs,
        )
        for i, caption in enumerate(captions):
            print(video_files[i])
            print(caption)
