import shutil
from pathlib import Path

def organize_nifti_files(main_folder):
    # Define paths
    main_path = Path(main_folder)
    dwi_folder = main_path / "DWIs"
    flair_folder = main_path / "FLAIRs"
    labels_folder = main_path / "Labels"
    
    # Create output folders if they don't exist
    dwi_folder.mkdir(exist_ok=True)
    flair_folder.mkdir(exist_ok=True)
    labels_folder.mkdir(exist_ok=True)
    
    # Counter for files processed
    dwi_count = 0
    flair_count = 0
    mask_count = 0
    
    # Walk through all patient folders
    for patient_folder in main_path.iterdir():
        if patient_folder.is_dir() and patient_folder.name not in ["DWIs", "FLAIRs", "Labels"]:
            # Search for DWI and mask files
            for file in patient_folder.glob('*.nii*'):  # Matches .nii and .nii.gz
                filename = file.name.lower()
                
                # Copy DWI files to Images folder
                if 'dwi' in filename:
                    dest = dwi_folder / f"{patient_folder.name}_{file.name}"
                    shutil.copy(file, dest)
                    dwi_count += 1
                    
                # Copy FLAIR files to Images folder
                if 'flair_reg' in filename:
                    dest = flair_folder / f"{patient_folder.name}_{file.name}"
                    shutil.copy(file, dest)
                    flair_count += 1
                
                # Copy mask files to Labels folder
                elif 'mask' in filename:
                    dest = labels_folder / f"{patient_folder.name}_{file.name}"
                    shutil.copy(file, dest)
                    mask_count += 1
    
    print(f"Organization complete!")
    print(f"Copied {dwi_count} DWI images to {dwi_folder}")
    print(f"Copied {flair_count} FLAIR images to {flair_folder}")
    print(f"Copied {mask_count} masks to {labels_folder}")

# Usage
if __name__=="__main__":

    dataset_path = "E:\\Codigos\\Python\\Brain Stroke Segmentation\\ISLE2022"

    organize_nifti_files(dataset_path)