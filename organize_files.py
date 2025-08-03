import os
import shutil
from pathlib import Path

def transform_bids_to_isle(bids_root, output_dir, mask_suffix='_msk'):
    """
    Convert BIDS dataset with session directories to ISLE2022-like structure.
    
    Args:
        bids_root (str): Path to BIDS dataset root directory
        output_dir (str): Path to output directory (will be created)
        mask_suffix (str): Suffix to identify segmentation masks
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Walk through BIDS directory
    for subject_dir in Path(bids_root).glob('sub-strokecase*'):
        if not subject_dir.is_dir():
            continue
            
        # Extract subject ID (e.g., 'strokecase0001' from 'sub-strokecase0001')
        sub_id = subject_dir.name.split('-')[1]
        patient_id = f"patient_{sub_id.replace('strokecase', '')}"
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(exist_ok=True)
        
        # Initialize variables for this subject
        dwi_path = None
        flair_path = None
        mask_path = None
        
        # Check all session directories
        for session_dir in subject_dir.glob('ses-*'):
            if not session_dir.is_dir():
                continue
                
            # Look for DWI in dwi directory
            dwi_dir = session_dir / 'dwi'
            if dwi_dir.exists():
                for dwi_file in dwi_dir.glob('*.nii.gz'):
                    if 'dwi' in dwi_file.name.lower():
                        dwi_path = dwi_file
                        break
            
            # Look for FLAIR in anat directory
            anat_dir = session_dir / 'anat'
            if anat_dir.exists():
                for anat_file in anat_dir.glob('*.nii.gz'):
                    if 'flair' in anat_file.name.lower():
                        flair_path = anat_file
                    elif mask_suffix in anat_file.name.lower():
                        mask_path = anat_file
        
        # Copy files to new structure
        if dwi_path:
            shutil.copy(dwi_path, patient_output_dir / 'dwi.nii.gz')
        if flair_path:
            shutil.copy(flair_path, patient_output_dir / 'flair.nii.gz')
        if mask_path:
            shutil.copy(mask_path, patient_output_dir / 'mask.nii.gz')
        
        # Check derivatives for masks if not found in main directory
        if not mask_path:
            derivatives_dir = Path(bids_root) / 'derivatives'
            if derivatives_dir.exists():
                deriv_sub_dir = derivatives_dir / subject_dir.name
                if deriv_sub_dir.exists():
                    for deriv_ses_dir in deriv_sub_dir.glob('ses-*'):
                        if deriv_ses_dir.exists():
                            for mask_file in deriv_ses_dir.glob(f'*{mask_suffix}.nii.gz'):
                                shutil.copy(mask_file, patient_output_dir / 'mask.nii.gz')
                                break
        
        print(f"Processed {patient_id}:")
        print(f"  DWI: {'found' if dwi_path else 'missing'}")
        print(f"  FLAIR: {'found' if flair_path else 'missing'}")
        print(f"  Mask: {'found' if (patient_output_dir / 'mask.nii.gz').exists() else 'missing'}")
    
    

if __name__ == '__main__':
    
    bids_directory = 'D:\\Trabajo\\Work\\Proyecto IA MRI Neurorreporte\\Datasets - MRI\\ISLES-2022\\ISLES-2022'
    output_directory = 'D:\\Trabajo\\Code\\Brain Stroke Segmentation\\ISLE2022'
    
    transform_bids_to_isle(bids_directory, output_directory)
    
    print("\nConversion complete!")
    print(f"BIDS dataset at {bids_directory} has been transformed to {output_directory}")
