# predict.py

import argparse
import os
import glob
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import random

# Albumentations for robust image perturbations
import albumentations as A
import cv2

# We must import the model's class definition to be able to load it.
from models.onnx_model import CSAD_ONNX

def set_seed(seed):
    """Sets a fixed random seed for reproducibility of perturbations."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    """
    Main function to run the inference script.
    """
    parser = argparse.ArgumentParser(
        description="Run inference on a folder of images using a trained CSAD model, with optional perturbations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images for inference.")
    
    # Arguments for perturbations
    parser.add_argument("--geometric", action="store_true", help="Apply geometric perturbations (perspective, rotation).")
    parser.add_argument("--color", action="store_true", help="Apply color-based perturbations (brightness, contrast).")
    parser.add_argument("--show_perturbations", action="store_true", help="Save side-by-side comparison images to a 'predictions' folder.")
    parser.add_argument("--save_perturbed", action="store_true", help="Save the perturbed images to a dynamically named subfolder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible perturbations.")

    args = parser.parse_args()

    # Set the seed for reproducible "random" augmentations
    set_seed(args.seed)

    # --- 1. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    print(f"Loading model from '{args.model_path}'...")
    state_dict = torch.load(args.model_path, map_location=device)

    try:
        num_classes = state_dict['segmentor.fc2.conv3.weight'].shape[0]
    except KeyError:
        print("Error: Could not determine num_classes from model file. Is this a valid CSAD model?")
        return

    model = CSAD_ONNX(dim=512, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Prepare Image Transformations ---
    # This is the standard preprocessing pipeline for the model.
    # It is applied *after* any perturbations.
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 4. Define Perturbation Pipeline using Albumentations ---
    perturbations = []
    perturbation_names = [] # For dynamic folder naming
    if args.geometric:
        print("Applying GEOMETRIC perturbations.")
        perturbations.append(A.Perspective(scale=(0.02, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=1.0))
        perturbations.append(A.SafeRotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=1.0))
        perturbation_names.append("geometric")
    
    if args.color:
        print("Applying COLOR perturbations.")
        perturbations.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0))
        # perturbations.append(A.GaussNoise(var_limit=(3.0, 5.0), p=1.0))
        perturbation_names.append("color")

    perturb_pipeline = A.Compose(perturbations) if perturbations else None

    # --- 5. Find and Process Images ---
    if not os.path.isdir(args.image_folder):
        print(f"Error: Image folder not found at '{args.image_folder}'")
        return

    image_paths = sorted(glob.glob(os.path.join(args.image_folder, '*.png')) + \
                         glob.glob(os.path.join(args.image_folder, '*.jpg')) + \
                         glob.glob(os.path.join(args.image_folder, '*.jpeg')))

    if not image_paths:
        print(f"No images found in '{args.image_folder}'.")
        return

    # --- Prepare output directories before the loop ---
    if args.show_perturbations and perturb_pipeline:
        comparison_output_dir = "predictions"
        os.makedirs(comparison_output_dir, exist_ok=True)
        print(f"Saving comparison images to '{comparison_output_dir}/'")

    if args.save_perturbed and perturb_pipeline:
        # Create a dynamic folder name based on the applied perturbations
        folder_suffix = "_".join(perturbation_names)
        perturbed_save_dir = os.path.join(args.image_folder, f"perturbed_{folder_suffix}")
        os.makedirs(perturbed_save_dir, exist_ok=True)
        print(f"Saving perturbed images to '{perturbed_save_dir}/'")

    print(f"\nFound {len(image_paths)} images. Starting inference...")
    print("-" * 60)
    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                filename = os.path.basename(img_path)
                image_pil = Image.open(img_path).convert("RGB")
                
                # --- Original Image Inference ---
                input_tensor_orig = preprocess(image_pil).unsqueeze(0).to(device)
                score_tensor_orig = model(input_tensor_orig)
                score_orig = score_tensor_orig.item()
                print(f"Image: {filename:<20} -> Original Score: {score_orig:8.4f}", end="")

                # --- Perturbed Image Inference (if requested) ---
                if perturb_pipeline:
                    # The correct pipeline:
                    # 1. Convert to NumPy format for perturbation.
                    image_np = np.array(image_pil)
                    # 2. Apply perturbations to the raw image data.
                    perturbed_data = perturb_pipeline(image=image_np)
                    image_perturbed_np = perturbed_data['image']
                    image_perturbed_pil = Image.fromarray(image_perturbed_np)
                    
                    # 3. Apply the standard preprocessing (including normalization) to the perturbed image.
                    input_tensor_pert = preprocess(image_perturbed_pil).unsqueeze(0).to(device)
                    score_tensor_pert = model(input_tensor_pert)
                    score_pert = score_tensor_pert.item()
                    print(f" | Perturbed Score: {score_pert:8.4f}")

                    # --- Updated saving logic ---
                    if args.show_perturbations:
                        image_pil_resized = image_pil.resize(image_perturbed_pil.size)
                        comparison_img = Image.new('RGB', (image_pil_resized.width * 2, image_pil_resized.height))
                        comparison_img.paste(image_pil_resized, (0, 0))
                        comparison_img.paste(image_perturbed_pil, (image_pil_resized.width, 0))
                        save_path = os.path.join(comparison_output_dir, f"comparison_{filename}")
                        comparison_img.save(save_path)
                    
                    if args.save_perturbed:
                        save_path = os.path.join(perturbed_save_dir, filename)
                        image_perturbed_pil.save(save_path)
                else:
                    print() # Newline if no perturbation

            except Exception as e:
                print(f"\nFailed to process {os.path.basename(img_path)}: {e}")
    
    print("-" * 60)
    print("Inference complete.")

if __name__ == "__main__":
    main()