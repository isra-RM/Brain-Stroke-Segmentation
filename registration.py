from pathlib import Path
import ants
from tqdm import tqdm  # Import tqdm

def coregister_img(folder_path):
    # Get list of subdirectories
    subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
    
    # Wrap subdirs with tqdm for progress bar
    for item in tqdm(subdirs, desc="Coregistering FLAIR to DWI..."):
        dwi_path = item/"dwi.nii.gz"
        flair_path = item/"flair.nii.gz"
        
        # Check if files exist before processing
        if not dwi_path.exists() or not flair_path.exists():
            print(f"Skipping {item.name}: Missing DWI or FLAIR")
            continue
        
        # Load images
        try:
            dwi = ants.image_read(str(dwi_path))
            flair = ants.image_read(str(flair_path))
        except Exception as e:
            print(f"Error loading {item.name}: {e}")
            continue

        # Coregister: FLAIR → DWI (Affine)
        tx = ants.registration(
            fixed=dwi,
            moving=flair,
            type_of_transform="Affine",
            interpolator="linear",
            metric="MutualInformation"
        )

        # Resample FLAIR to DWI space
        flair_reg = ants.apply_transforms(
            fixed=dwi,
            moving=flair,
            transformlist=tx["fwdtransforms"],
            interpolator="linear"
        )

        # Save resampled FLAIR
        output_path = item/"flair_reg.nii.gz"
        flair_reg.to_file(str(output_path))

if __name__ == "__main__":
    
    folder_path = Path("D:\\Trabajo\\Code\\Brain Stroke Segmentation\\ISLE2022")
    coregister_img(folder_path)