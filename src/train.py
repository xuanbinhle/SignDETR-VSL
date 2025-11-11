from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from torch.utils.data import DataLoader 
from torch import optim, load, save
from colorama import Fore 
from utils.logger import get_logger
from utils.rich_handlers import TrainingHandler, rich_training_context
import sys 
import torch
from utils.boxes import stacker
from torch.amp import GradScaler, autocast  # Import for Mixed Precision

if __name__ == '__main__': 
    # Initialize logger and handlers
    logger = get_logger("training")
    logger.print_banner()
    
    # --- OPTIMIZATION 1: Increase Batch Size & add num_workers ---
    # 8754 train files / batch_size 32 = 273 batches per epoch
    NEW_BATCH_SIZE = 16
    # Set num_workers to 4 or 8 to speed up data loading
    NUM_WORKERS = 2 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = False # Default to false
    if device.type == 'cpu':
        logger.warning("CUDA not available. Training on CPU.")
        # AMP (autocast/scaler) is not supported on CPU
    else:
        logger.info(f"Training on: {torch.cuda.get_device_name(0)}")
        use_amp = True

    train_dataset = DETRData('combineddata/train') 
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=NEW_BATCH_SIZE, 
                                  collate_fn=stacker, 
                                  drop_last=True, 
                                  num_workers=NUM_WORKERS, 
                                  pin_memory=True) 

    # --- FIX: Pass only the root 'combineddata' path ---
    test_dataset = DETRData('combineddata/val', train=False) 
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=NEW_BATCH_SIZE, 
                                 collate_fn=stacker, 
                                 drop_last=True, 
                                 num_workers=NUM_WORKERS, 
                                 pin_memory=True) 

    num_classes = 128 
    model = DETR(num_classes=num_classes)
    # model.load_pretrained('pretrained/4426_model.pt')
    # model.log_model_info()
    model = model.to(device)
    model.train() 

    # --- OPTIMIZATION 2: Adjust Learning Rate ---
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataloader)*30, T_mult=2)
    weights= {'class_weighting': 5, 'bbox_weighting': 3, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)
    criterion.to(device)

    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    epochs = 10000
    
    # --- OPTIMIZATION 3: Add GradScaler for Mixed Precision ---
    scaler = GradScaler(enabled=use_amp)
    
    # Log training configuration
    training_config = {
        "Total Epochs": epochs,
        "Batch Size": NEW_BATCH_SIZE, # Updated
        "Train Batches": train_batches,
        "Test Batches": test_batches,
        "Learning Rate": "1e-4 (Adjusted)", # Updated
        "Optimizer": "Adam",
        "Scheduler": "CosineAnnealingWarmRestarts",
        "Mixed Precision": f"{use_amp} (AMP)" # Updated
    }
    logger.print_table("🏋️ Training Configuration", list(training_config.keys()), [list(training_config.values())])
    
    # Start training with rich context
    with rich_training_context() as training_handler:
        for epoch in range(epochs): 
            with training_handler.create_training_progress() as epoch_progress:
                epoch_task = epoch_progress.add_task(f"[bold blue] Progress {epoch+1}/{epochs}", train_loss=0.0, test_loss=0.0, total=train_batches)
                # Training phase
                model.train()
                train_epoch_loss = 0.0 
            
                # Create progress bar for current epoch
                for batch_idx, batch in enumerate(train_dataloader): 
                    X, y = batch
                    X = X.to(device)
                    y = [{k: v.to(device) for k, v in t.items()} for t in y]
                    try: 
                        # --- OPTIMIZATION 3.1: Apply autocast ---
                        # autocast runs the forward pass in mixed precision
                        with autocast(device_type=device.type, enabled=use_amp):
                            yhat = model(X) 
                            yhat_classes = yhat['pred_logits'] 
                            yhat_bb = yhat['pred_boxes'] 
                            loss_dict = criterion(yhat, y) 
                            weight_dict = criterion.weight_dict
                            
                            # Ensure we sum exactly over the expected weighted keys, and keep tensor dtype
                            losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                        
                        # Calculate loss 
                        train_epoch_loss += losses.item() 
                        
                        # Zero grads
                        opt.zero_grad()
                        
                        # --- OPTIMIZATION 3.2: Use scaler ---
                        if use_amp:
                            scaler.scale(losses).backward()
                            scaler.step(opt)
                            scaler.update()
                        else:
                            # Standard backward pass if on CPU
                            losses.backward()
                            opt.step()
                        
                        # Update progress
                        epoch_progress.update(epoch_task, advance=1, train_loss=round(train_epoch_loss/train_batches,5))
                        
                        # Update progress
                        epoch_progress.update(epoch_task, advance=1, train_loss=round(train_epoch_loss/train_batches,5))
                        
                    except Exception as e: 
                        logger.error(f"Training error at epoch {epoch}, batch {batch_idx}: {str(e)}")
                        logger.error(f"Batch targets: {str(y)}")
                        sys.exit()
                
                # Progress lr 
                scheduler.step()
            
                # Test phase
                model.eval()
                test_epoch_loss = 0.0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_dataloader):
                        X, y = batch
                        X = X.to(device)
                        y = [{k: v.to(device) for k, v in t.items()} for t in y]
                        # --- OPTIMIZATION 3.3: Use autocast in validation ---
                        with autocast(device_type=device.type, enabled=use_amp):
                            yhat = model(X)
                            loss_dict = criterion(yhat, y) 
                            weight_dict = criterion.weight_dict
                            losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                        
                        # Calculate loss 
                        test_epoch_loss += losses.item() 
                        epoch_progress.update(epoch_task, advance=0, test_loss=round(test_epoch_loss/test_batches,5))
                
                # Save checkpoints
                if epoch % 10 == 0 and epoch != 0: 
                    checkpoint_path = f"checkpoints/{epoch}_model.pt"
                    save(model.state_dict(), checkpoint_path)
                    training_handler.save_checkpoint_status(checkpoint_path, epoch)
            
    # Final save
    save(model.state_dict(), f"checkpoints/{epoch}_model.pt")