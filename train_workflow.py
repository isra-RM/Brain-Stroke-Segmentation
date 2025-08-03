import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
from pathlib import Path
import logging
import torch
from typing import Dict, List, Optional, Tuple
from monai.losses import DiceLoss 
from monai.transforms import (
    Compose, NormalizeIntensityd, AsDiscreted, LoadImaged, EnsureChannelFirstd,Spacingd,
    Resized, Rand3DElasticd, RandFlipd, RandRotate90d, Rand3DElasticd,RandBiasFieldd,SpatialPadd,
    RandShiftIntensityd, EnsureTyped, Activationsd, EnsureTyped, KeepLargestConnectedComponentd
)
from monai.networks.nets import UNet, SegResNet # Changed to UNet
from monai.data import Dataset, DataLoader
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler, TensorBoardStatsHandler,TensorBoardImageHandler,EarlyStopHandler,
    ValidationHandler, CheckpointSaver, CheckpointLoader,MeanDice,LrScheduleHandler
)
from ignite.metrics import Accuracy
from monai.handlers.utils import from_engine
from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.utils import set_determinism
from monai.config import print_config
from monai.apps import get_logger

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class StrokeSegmentationWorkflow:
    
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        chkpt_dir: str,
        max_epochs: int = 100,
        device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 2,
        num_workers: int = 2,
    ):
        """
        Stroke segmentation workflow for training and evaluation
        
        Args:
            dataset_dir: Directory containing dataset
            output_dir: Directory for output files and logs
            chkpt_dir: Directory for saving checkpoints
            max_epochs: Maximum training epochs
            device: 'cuda', 'cpu'
            spatial_size: Target spatial size for resizing
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
        """
        get_logger("train_log")
        print_config()
        set_determinism(seed=123)
        
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.chkpt_dir = chkpt_dir
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        # Create directories with pathlib
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(chkpt_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_model()
        self._initialize_loss()
        self._initialize_optimizer()
        self._initialize_key_metric()
        logger.info(f"Workflow initialized for ischemic brain stroke segmentation!")
        
        logger.info(f"Workflow initialized!")

        
    def _initialize_model(self):
        """Initialize segmentation model"""
        self.model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2
        ).to(self.device)
        
    def _initialize_loss(self):
        """Initialize loss function"""
        self.loss = DiceLoss(
            sigmoid=True,
            squared_pred=True,
            smooth_nr=0,
            smooth_dr=1e-5
        )
        
    def _initialize_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.max_epochs
        )
        
    def _initialize_key_metric(self):
        self.key_metric = MeanDice(
            include_background=True,
            reduction="mean",
            output_transform=from_engine(["pred", "label"])
        )
    
    def create_datalist(self, start_idx: int, end_idx: int) -> List[Dict]:
        """Create file list for given index range"""
        # Convert to Path objects
        dataset_path = Path(self.dataset_dir)
        image_dir = dataset_path/'DWIs'
        label_dir = dataset_path/'Labels'
        
        # Get sorted lists of NIfTI files using pathlib
        images = sorted(image_dir.glob('*.nii.gz'))
        labels = sorted(label_dir.glob('*.nii.gz'))
        
        # Verify we found files
        if not images:
            raise FileNotFoundError(f"No images found in {image_dir}")
        if not labels:
            raise FileNotFoundError(f"No labels found in {label_dir}")
        
        # Check for mismatched counts
        if len(images) != len(labels):
            raise ValueError(f"Mismatched files: {len(images)} images vs {len(labels)} labels")
        
        # Create dictionary list with string paths
        datalist = [
            {'image': str(img), 'label': str(lbl)} 
            for img, lbl in zip(images, labels)
        ]
        
        # Validate indices
        total = len(datalist)
        if start_idx < 0 or end_idx >= total or start_idx > end_idx:
            raise IndexError(
                f"Invalid indices {start_idx}-{end_idx} for dataset size {total}"
            )
        
        return datalist[start_idx : end_idx + 1]
            
    def create_transforms(self, augment: bool = False) -> Compose:
        """Create data transformation pipeline"""
        det_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],spatial_size=(176,176,64),mode=("trilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
        ])
        
        if augment:
            prob = 0.5
            rand_transforms = Compose([
                #RandSpatialCropd(keys=["image", "label"],roi_size=(176,176,64),random_size=False),
                RandFlipd(keys=["image", "label"], prob=prob, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=prob, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=prob, spatial_axis=2),
                RandRotate90d(keys=["image", "label"],max_k=3, prob=prob),
                #Rand3DElasticd(keys=["image", "label"], sigma_range=(5,7),magnitude_range=(50,100),padding_mode='zeros',prob=0.2, mode=("trilinear","nearest")),
                RandBiasFieldd(keys=["image"],coeff_range=(0.1,0.2),prob=prob),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=prob)
            ])
            
            return Compose([det_transforms,rand_transforms])
        else:
            return det_transforms
    
    def create_postprocessing(self) -> Compose:
        """Create postprocessing pipeline"""
        
        post_transforms = Compose([
            EnsureTyped(keys=["pred", "label"], data_type="tensor"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", connectivity=2)
        ])
        
        return post_transforms
    
    def prepare_datasets(self, train_range: Tuple[int, int], val_range: Tuple[int, int]):
        """
        Prepare datasets with proper data splitting
        
        Args:
            train_range: (start, end) indices for training data
            val_range: (start, end) indices for validation data
        """
        logger.info("Preparing datasets...")
        train_files = self.create_datalist(*train_range)
        val_files = self.create_datalist(*val_range)
        
        self.train_preprocessing = self.create_transforms(augment=True)
        self.val_preprocessing = self.create_transforms(augment=False)
        self.postprocessing = self.create_postprocessing()
        
        self.train_dataset = Dataset(
            data=train_files,
            transform=self.train_preprocessing,
        )
        self.val_dataset = Dataset(
            data=val_files,
            transform=self.val_preprocessing, 
        )
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} images for training, {len(self.val_dataset)} images for validation")
            
    def prepare_dataloaders(self):
        logger.info("Preparing dataloaders...")
        """Prepare data loaders"""
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )    

    
    def prepare_training(self, resume_checkpoint: Optional[str] = None):
        
        if resume_checkpoint is not None and Path(resume_checkpoint).exists():
            logger.info(f"Resuming training from {resume_checkpoint}...")
        else:
            logger.info("Setting up training engines...") 
        
        train_inferer = SimpleInferer()
        #val_inferer = SlidingWindowInferer(roi_size=(196,196,80), sw_batch_size=1, overlap=0.5)
        val_inferer = SimpleInferer()
        
        self.evaluator = SupervisedEvaluator(
        device=self.device,
        val_data_loader=self.val_dataloader,
        network=self.model,
        inferer=val_inferer,
        key_val_metric={"val_mean_dice":self.key_metric},
        postprocessing=self.postprocessing,
        val_handlers= [
            StatsHandler(name="train_log", output_transform=lambda x: None),
            TensorBoardStatsHandler(log_dir=self.output_dir, output_transform=lambda x: None,),
            TensorBoardImageHandler(
            log_dir=self.output_dir,
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
            CheckpointSaver(
                save_dir=self.chkpt_dir,
                save_dict={"model": self.model},
                save_key_metric=True,
                key_metric_filename="best_model.pt",
                )
            ] ,
        amp=True       
        )   
        
        self.trainer = SupervisedTrainer(
        device=self.device,
        max_epochs=self.max_epochs,
        train_data_loader=self.train_dataloader,
        network=self.model,
        optimizer=self.optimizer,
        loss_function=self.loss,
        inferer=train_inferer,
        postprocessing=self.postprocessing,
        key_train_metric={"train_mean_dice":self.key_metric},
        train_handlers = [
            ValidationHandler(validator=self.evaluator, epoch_level=True, interval=1),
            LrScheduleHandler(lr_scheduler=self.lr_scheduler, print_lr=True),
            StatsHandler(name="train_log",tag_name="train_loss", output_transform=from_engine(['loss'], first=True)),
            TensorBoardStatsHandler(log_dir=self.output_dir,tag_name="train_loss", output_transform=from_engine(['loss'], first=True))
        ],
        amp=True
        )
        
        early_stopper = EarlyStopHandler(
            patience=20,
            score_function=lambda x: x.state.metrics["val_mean_dice"],
            epoch_level=True,
            trainer=self.trainer,
            min_delta=0.05       
        )
        
        early_stopper.attach(self.evaluator)
        
        checkpoint_saver = CheckpointSaver(
            save_dir=self.chkpt_dir,
            save_dict={
                "model": self.model,
                "optimizer": self.optimizer,
                "trainer": self.trainer,
                "scheduler": self.lr_scheduler 
            },
            save_interval=1,  # Save every epoch
            epoch_level=True,
            save_final=False,  
            save_key_metric=False,
            n_saved=1
        )
        
        checkpoint_saver.attach(self.trainer)
        
        if resume_checkpoint is not None and Path(resume_checkpoint).exists():
            
            checkpoint_loader = CheckpointLoader(
            load_path=resume_checkpoint,
            load_dict={
                "model": self.model,
                "optimizer": self.optimizer,
                "trainer": self.trainer,
                "scheduler": self.lr_scheduler 
            },
            map_location=self.device
            )
        
            checkpoint_loader.attach(self.trainer) 
    
    def load_pretrained_weights(
        self,
        pretrained_path: str,
        freeze_layers: Optional[int] = None
    ):
        """
        Load pretrained weights with channel adaptation
        
        Args:
            pretrained_path: Path to pretrained weights
            freeze_layers: Number of initial layers to freeze
        """
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        
        if pretrained_path is not None and Path(pretrained_path).exists():
            
            pretrained_dict = torch.load(pretrained_path, map_location=self.device)
            model_dict = self.model.state_dict()
            
            # Handle input channel mismatch
            #if 'convInit.conv.weight' in pretrained_dict:
            # Pretrained has 4 input channels, we have 1
            init_weights = pretrained_dict['convInit.conv.weight']
            out_weights = pretrained_dict['conv_final.2.conv.weight'] 
            out_bias = pretrained_dict['conv_final.2.conv.bias']  
            # Option 1: Use mean of all input channels
            adapted_init_weights = init_weights.mean(dim=1, keepdim=True)
            adapted_out_weights = out_weights.mean(dim=0, keepdim=True)
            adapted_out_bias = out_bias.mean(dim=0, keepdim=True)   
            # Option 2: Use just the first channel (if you know it's most relevant)
            # adapted_weights = pretrained_weights[:, :1, :, :, :].clone()
            pretrained_dict['convInit.conv.weight'] = adapted_init_weights
            pretrained_dict['conv_final.2.conv.weight'] = adapted_out_weights
            pretrained_dict['conv_final.2.conv.bias'] = adapted_out_bias
                
            # 1. Filter out unnecessary keys (mismatched layers)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                if k in model_dict and v.shape == model_dict[k].shape}
                
            # 2. Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
                
            # 3. Load the modified state dict
            self.model.load_state_dict(model_dict, strict=False)  # strict=False allows partial loading
            
            logger.info(
                f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers "
                f"({len(pretrained_dict)/len(model_dict):.1%})"
            )
            
            # Freeze layers if requested
            if freeze_layers is not None:
                self.freeze_layers(freeze_layers)

            else:
                raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
        

    def freeze_layers(self, num_layers: int):
        """Freeze first N layers of the model"""
        logger.info(f"Freezing first {num_layers} layers")
        layers_frozen = 0
        
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i < num_layers:
                param.requires_grad = False
                layers_frozen += 1
                logger.debug(f"Frozen layer: {name}")
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Frozen {layers_frozen} layers. "
            f"Trainable parameters: {trainable}/{total} ({trainable/total:.1%})"
        )
               
    def train(
        self,
        train_indices: Tuple[int, int] = (0, 200),
        val_indices: Tuple[int, int] = (201, 250),
        pretrained_path: Optional[str] = None,
        freeze_layers: Optional[int] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Run training workflow
        
        Args:
            train_indices: (start, end) patient indices for training
            val_indices: (start, end) patient indices for validation
            pretrained_path: Path to pretrained weights
            freeze_layers: Number of layers to freeze
            resume: Checkpoint path for resuming training
        """
        # Prepare data
        self.prepare_datasets(train_indices, val_indices)
        self.prepare_dataloaders()
        
        # Handle saevd checkpoint   
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self.prepare_training(resume_checkpoint=checkpoint_path)
        # Handle pretrained weights
        elif pretrained_path is not None and Path(pretrained_path).exists():
            self.load_pretrained_weights(pretrained_path, freeze_layers)
            self.prepare_training(resume_checkpoint=None)
        # Handle training from scratch    
        else:
            self.prepare_training(resume_checkpoint=None)
        
        # Start training
        logger.info("Training started...")
        self.trainer.run()
        logger.info("Training completed!")
               
    
if __name__ == "__main__":
        
    dataset_dir = 'E:\\Codigos\\Python\\Brain Stroke Segmentation\\ISLE2022'
    output_dir = 'E:\\Codigos\\Python\\Brain Stroke Segmentation\\results'
    chkpt_dir = 'E:\\Codigos\\Python\\Brain Stroke Segmentation\\models'

    pretrained_path = 'E:\\Codigos\\Python\\Brain Stroke Segmentation\\pretrained_model.pt' 
        
    workflow = StrokeSegmentationWorkflow(dataset_dir=dataset_dir,output_dir=output_dir,chkpt_dir=chkpt_dir)
    # Run training
    workflow.train(train_indices=(0, 200),val_indices=(201, 249),pretrained_path=pretrained_path,freeze_layers=0)
