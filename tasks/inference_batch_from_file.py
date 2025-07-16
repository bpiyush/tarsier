"""Inference script to caption a batch of videos from a file list."""
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
import json
import argparse
import glob
from pathlib import Path


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


def load_video_list(file_path):
    """Load list of video paths from a text file."""
    with open(file_path, 'r') as f:
        video_files = [line.strip() for line in f if line.strip()]
    return video_files


def get_existing_captions(save_dir):
    """Scan save_dir and return set of video paths that already have captions."""
    existing_videos = set()
    
    if not os.path.exists(save_dir):
        return existing_videos
    
    # Look for JSON files in save_dir
    json_files = glob.glob(os.path.join(save_dir, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                existing_videos.update(data.keys())
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not read {json_file}")
            continue
    
    return existing_videos


def save_captions_batch(captions_dict, save_dir, batch_num, num_save):
    """Save a batch of captions to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Split captions_dict into chunks of num_save
    items = list(captions_dict.items())
    for i in range(0, len(items), num_save):
        chunk = dict(items[i:i + num_save])
        chunk_num = batch_num + (i // num_save)
        output_file = os.path.join(save_dir, f"captions_batch_{chunk_num:04d}.json")
        
        with open(output_file, 'w') as f:
            json.dump(chunk, f, indent=2)
        
        print(f"Saved {len(chunk)} captions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch video captioning from file list")
    parser.add_argument("--file", type=str, required=True, 
                       help="Text file containing list of video paths")
    parser.add_argument("--batch_size", type=int, default=12,
                       help="Batch size for processing (default: 12)")
    parser.add_argument("--num_save", type=int, required=True,
                       help="Number of samples to save in each output file")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Directory to save caption JSON files")
    parser.add_argument("--config", type=str, default="configs/tarser2_default_config.yaml",
                       help="Path to config file")
    parser.add_argument("--model_path", type=str, 
                       default="/work/piyush/pretrained_checkpoints/Tarsier2/Tarsier2-Recap-7b/",
                       help="Path to model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for generation (default: 0.0)")
    parser.add_argument("--prompt", type=str, default="Describe the video in detail.",
                       help="Prompt for captioning")
    
    args = parser.parse_args()
    
    # Load video list
    print(f"Loading video list from {args.file}")
    all_video_files = load_video_list(args.file)
    print(f"Found {len(all_video_files)} videos in file")
    
    # Check for existing captions
    print(f"🔍 Scanning {args.save_dir} for existing captions...")
    existing_videos = get_existing_captions(args.save_dir)
    print(f"✅ Found {len(existing_videos)} videos with existing captions")
    
    # Filter out videos that already have captions
    video_files_to_process = [v for v in all_video_files if v not in existing_videos]
    print(f"Will process {len(video_files_to_process)} new videos")
    
    if len(video_files_to_process) == 0:
        print("No new videos to process. Exiting.")
        return
    
    # Setup generation parameters
    generate_kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "max_new_tokens": 256,
        "top_p": 1,
        "temperature": args.temperature,
        "use_cache": True
    }
    
    # Load model and processor
    print("Loading model and processor...")
    data_config = yaml.safe_load(open(args.config, 'r'))
    model, processor = load_model_and_processor(
        args.model_path, data_config=data_config,
    )
    
    # Process videos in batches
    all_captions = {}
    batch_num = 0
    total_batches = (len(video_files_to_process) + args.batch_size - 1) // args.batch_size
    
    print(f"Processing {len(video_files_to_process)} videos in {total_batches} batches of {args.batch_size}...")
    
    # Simple progress bar
    for i in tqdm(range(0, len(video_files_to_process), args.batch_size), 
                 desc="Processing videos", unit="batch"):
        batch_videos = video_files_to_process[i:i + args.batch_size]
        current_batch = i // args.batch_size + 1
        
        try:
            captions = process_multiple_videos(
                model, processor, args.prompt, batch_videos, generate_kwargs
            )
            
            # Store captions
            for video_path, caption in zip(batch_videos, captions):
                all_captions[video_path] = caption
            
            # Save batch if we have enough samples
            if len(all_captions) >= args.num_save:
                save_captions_batch(all_captions, args.save_dir, batch_num, args.num_save)
                all_captions = {}
                batch_num += 1
                
        except Exception as e:
            print(f"\nError processing batch {current_batch}: {e}")
            print(f"Videos in failed batch: {batch_videos}")
            continue
    
    # Save remaining captions
    if all_captions:
        save_captions_batch(all_captions, args.save_dir, batch_num, args.num_save)
    
    print("\n🎉 Processing complete!")
    print(f"📊 Summary:")
    print(f"   • Total videos processed: {len(video_files_to_process)}")
    print(f"   • Captions saved: {len(video_files_to_process)}")
    print(f"   • Output directory: {args.save_dir}")
    print(f"   • Batch size used: {args.batch_size}")
    print(f"   • Samples per file: {args.num_save}")


if __name__ == "__main__":
    main() 