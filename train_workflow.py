import os
import torch
from monai.losses import DiceLoss
from monai.transforms import (Compose,NormalizeIntensityd,
    AsDiscreted,LoadImaged,EnsureChannelFirstd, DeleteItemsd,ConcatItemsd,
    Activationsd, Resized, RandSpatialCropd,RandFlipd,RandRotated,
    RandScaleIntensityd,RandShiftIntensityd,EnsureTyped
)
from monai.losses import DiceLoss
from monai.networks.nets import SegResNet
from monai.data import CacheDataset,Dataset,DataLoader,decollate_batch
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    LrScheduleHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    CheckpointSaver,
    CheckpointLoader,
)
from monai.handlers.utils import from_engine
from monai.handlers import MeanDice
from monai.inferers import SlidingWindowInferer,SimpleInferer
from monai.utils import set_determinism
from monai.config import print_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrokeWorkflow:
    
    def __init__(self, dataset_dir, output_dir, chkpt_dir, max_epochs=200):
        print_config()
        set_determinism(seed=123)
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.chkpt_dir = chkpt_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.model = SegResNet(
                    spatial_dims=3,
                    in_channels=1,  # DWI + FLAIR
                    out_channels=1,  # Binary output
                    init_filters=16,
                    blocks_down=(1, 2, 2, 4),
                    blocks_up=(1, 1, 1),
                    dropout_prob=0.2
                ).to(self.device)
        self.max_epochs = max_epochs
        self.loss = DiceLoss(sigmoid=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-5)
        self.key_metric = MeanDice(include_background=True, reduction="mean",output_transform=from_engine(["pred", "label"]))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)
    
    def prepare_datasets(self):
        logger.info("Preparing datasets...")
        # Load data lists
        train_files = [
                        {
                            "image": os.path.join(self.dataset_dir, f"patient_{i:04d}\\dwi.nii.gz"),
                            #"flair": os.path.join(dataset_dir, f"patient_{i:04d}\\flair_reg.nii.gz"),
                            "label": os.path.join(self.dataset_dir, f"patient_{i:04d}\\mask.nii.gz")
                        } 
                        for i in range(1,201)  
                ]
                    
        val_files = [
                        {
                            "image": os.path.join(self.dataset_dir, f"patient_{i:04d}\\dwi.nii.gz"),
                            #"flair": os.path.join(dataset_dir, f"patient_{i:04d}\\flair_reg.nii.gz"),
                            "label": os.path.join(self.dataset_dir, f"patient_{i:04d}\\mask.nii.gz")
                        } 
                        for i in range(201,251)  
                    ]
        
        self.set_transforms()
        
        self.train_dataset = CacheDataset(data=train_files, transform=self.train_preprocessing, cache_rate=0.5)
        self.val_dataset = CacheDataset(data=val_files, transform=self.val_preprocessing, cache_rate=0.5)
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} validation")
            

    def set_transforms(self):
        
        target_size = (256,256,96)
        
        deterministic_transforms = Compose([
                            LoadImaged(keys=["image","label"]),
                            EnsureChannelFirstd(keys=["image","label"]),
                            Resized(keys=["image","label"],
                                    spatial_size = target_size,
                                    mode = ("trilinear","nearest")               
                            ),
                            #ConcatItemsd(keys=["dwi", "flair"],name="image",dim=0),
                            #DeleteItemsd(keys=["dwi", "flair"]),
                            NormalizeIntensityd(keys=["image"], nonzero = True, channel_wise=True)          
                            ])
        
        random_transforms = Compose([
                            RandSpatialCropd(
                                keys=["image", "label"],
                                roi_size=target_size,
                                random_size = False
                            ),
                            RandFlipd(
                                keys=["image", "label"],
                                prob = 0.5,
                                spatial_axis=0
                                
                            ),
                            RandFlipd(
                                keys=["image", "label"],
                                prob = 0.5,
                                spatial_axis=1
                                
                            ),
                            RandFlipd(
                                keys=["image", "label"],
                                prob = 0.5,
                                spatial_axis=2
                                
                            ),
                            RandRotated(
                                keys=["image", "label"], 
                                range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5
                            ),
                            RandScaleIntensityd(
                                keys=["image"], 
                                factors=0.1,
                                prob=1.0
                            ),
                            RandShiftIntensityd(
                                keys=["image"], 
                                offsets=0.1,
                                prob=1.0
                            )
                        ])
        post_transforms = Compose([
                EnsureTyped(keys=["pred","label"], data_type="tensor"),
                Activationsd(keys="pred",sigmoid=True),
                AsDiscreted(keys="pred",threshold=0.5)
            ])
        
        self.train_preprocessing = Compose([
            deterministic_transforms,
            random_transforms
            ])
        
        self.val_preprocessing = deterministic_transforms
        
        self.train_postprocessing = post_transforms
        self.val_postprocessing = post_transforms
        
    def prepare_dataloaders(self):
        
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=2,
                                           shuffle=True,
                                           num_workers=2)
        
        self.val_dataloader = DataLoader(self.val_dataset, 
                                           batch_size=2,
                                           shuffle=False,
                                           num_workers=2)
    
    def prepare_training(self, resume_checkpoint = None):
        
        logger.info("Setting up training engines...") 
        
        train_inferer = SimpleInferer()
        val_inferer = SlidingWindowInferer(roi_size=(256,256,96), sw_batch_size=1, overlap=0.5)
        
        
        val_handlers = [
            StatsHandler(iteration_log=False),
            TensorBoardStatsHandler(log_dir=self.output_dir,iteration_log=False),
            CheckpointSaver(
                save_dir=self.chkpt_dir,
                save_dict={
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "lr_scheduler":self.lr_scheduler
                    },
                save_key_metric=True,
                key_metric_filename="best_model.pt",
                ),
            ],     
        
        self.evaluator = SupervisedEvaluator(
        device=self.device,
        val_data_loader=self.val_dataloader,
        network=self.model,
        inferer=val_inferer,
        key_val_metric={"val_dice":self.key_metric},
        postprocessing=self.val_postprocessing,
        val_handlers= val_handlers     
        )   
        
        train_handlers = [
            LrScheduleHandler(lr_scheduler=self.lr_scheduler,print_lr=True),
            ValidationHandler(validator=self.evaluator, epoch_level=True, interval=1),
            StatsHandler(tag_name="train_loss", output_transform=from_engine(['loss'], first=True)),
            TensorBoardStatsHandler(log_dir=self.output_dir,tag_name="train_loss", output_transform=from_engine(['loss'], first=True)),
            CheckpointSaver(
                save_dir=self.chkpt_dir,
                save_dict={
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "lr_scheduler": self.lr_scheduler,
                    "trainer": self.trainer  # Will save state_dict automatically
                },
                save_interval=1,  # Save every epoch
                epoch_level=True,
                save_final=False,  
                filename_prefix="checkpoint_epoch_{epoch}",
                save_key_metric=False
            )
        ]
        
        if resume_checkpoint is not None:
            logger.info(f"Adding checkpoint loader for resuming training from {resume_checkpoint}")
            train_handlers.insert(0,
                CheckpointLoader(
                    load_path=resume_checkpoint,
                    load_dict={
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "lr_scheduler":self.lr_scheduler,
                    "trainer": self.trainer 
                    },
                    map_location=self.device
                )
            )
        
        self.trainer = SupervisedTrainer(
        device=self.device,
        max_epochs=self.max_epochs,
        train_data_loader=self.train_dataloader,
        network=self.model,
        optimizer=self.optimizer,
        loss_function=self.loss,
        inferer=train_inferer,
        postprocessing=self.train_postprocessing,
        key_train_metric={"train_dice":self.key_metric},
        train_handlers=train_handlers
        )
    
    def freeze_first_n_layers(self, n=None):
        
        logger.info(f"Freezing first {n} layers for fine-tuning")
        
        # Get all named parameters
        total_params = list(self.model.named_parameters())
        layers_frozen = 0
        
        # Freeze the first N layers
        for i, (name, param) in enumerate(total_params):
            if i < n:
                param.requires_grad = False
                layers_frozen += 1
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Frozen {layers_frozen} layers. Trainable parameters: "
                    f"{trainable_params}/{total_params} ({trainable_params/total_params:.1%})")
    
        
    def load_pretrained_model(self, pretrained_model_path, num_layers_freze=None):
        
        logger.info(f"Loading pretrained weights from {pretrained_model_path}")
        pretrained_dict = torch.load(pretrained_model_path, map_location=self.device)
        model_dict = self.model.state_dict()
            
        # Special handling for first conv layer (input channel mismatch)
        if 'convInit.conv.weight' in pretrained_dict:
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
                
            # Option 3: Randomly select one channel
            # import random
            # channel_idx = random.randint(0, 3)
            # adapted_weights = pretrained_weights[:, channel_idx:channel_idx+1, :, :, :].clone()
                
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
            
            logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
        
        # Freeze layers if requested
        if num_layers_freze is not None:
            self.freeze_first_n_layers(num_layers_freze)
               
    def run(self, pretrained_model_path,num_layers_freze):
        
        self.prepare_datasets()
        self.prepare_dataloaders()
        
        if pretrained_model_path is not None:
            self.load_pretrained_model(pretrained_model_path,num_layers_freze)
            
        self.prepare_training()
        self.trainer.run()

    def resume_training(self, checkpoint_path):
      
        self.prepare_datasets()
        self.prepare_dataloaders()
        
        self.prepare_training(resume_checkpoint=checkpoint_path)
        self.trainer.run()
        
        
    
if __name__=="__main__":
             
    dataset_dir = 'D:\\Trabajo\\Code\\Brain Stroke Segmentation\\ISLE2022'
    output_dir = 'D:\\Trabajo\\Code\\Brain Stroke Segmentation\\results'
    chkpt_dir = 'D:\\Trabajo\\Code\\Brain Stroke Segmentation\\models'
    
        
    stroke_segm = StrokeWorkflow(dataset_dir=dataset_dir,output_dir=output_dir,chkpt_dir=chkpt_dir,max_epochs=50)
    
    #stroke_segm.run(pretrained_model_path='pretrained_model.pt',num_layers_freze=0)
    stroke_segm.resume_training(checkpoint_path=os.path.join(chkpt_dir,'model.pt'))

