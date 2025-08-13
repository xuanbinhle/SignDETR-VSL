from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from torch.utils.data import DataLoader 
from torch import optim, save
from colorama import Fore  
import sys 
import torch
from utils.boxes import stacker

if __name__ == '__main__': 
    train_dataset = DETRData('data/train') 
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    test_dataset = DETRData('data/test', train=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    num_classes = 3 
    model = DETR(num_classes=num_classes)
    # model.load_state_dict(load('archive/goodcheckpoints/1000_model.pt'))
    model.train() 

    opt = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataloader)*30, T_mult=2)

    weights= {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)

    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    epochs = 5000
    
    for epoch in range(epochs): 
        # Training phase
        model.train()
        train_epoch_loss = 0.0 
        
        for batch_idx, batch in enumerate(train_dataloader): 
            X, y = batch
            try: 

                yhat = model(X) 
                # print(Fore.LIGHTBLUE_EX + 'predictions made' + Fore.RESET) 
                yhat_classes = yhat['pred_logits'] 
                yhat_bb = yhat['pred_boxes'] 
                # print(Fore.LIGHTBLUE_EX + 'loss calc starting' + Fore.RESET) 
                loss_dict = criterion(yhat, y) 
                # print(Fore.LIGHTBLUE_EX + 'loss calculated' + Fore.RESET) 
                weight_dict = criterion.weight_dict
                # Ensure we sum exactly over the expected weighted keys, and keep tensor dtype
                losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                
                # Calculate loss 
                train_epoch_loss += losses.item() 
                
                # Zero grads
                opt.zero_grad()
                # print(Fore.LIGHTBLUE_EX + 'grads zeroed' + Fore.RESET) 
                
                # Backward
                losses.backward()
                # print(Fore.LIGHTBLUE_EX + 'grads calculated' + Fore.RESET) 
                # Apply
                opt.step()
                # print(Fore.LIGHTBLUE_EX + 'grads applied' + Fore.RESET) 
                
                
            except Exception as e: 
                print(Fore.LIGHTYELLOW_EX + str(e) + Fore.RESET) 
                print(Fore.LIGHTRED_EX + str(y) + Fore.RESET) 
                sys.exit()

            # Fancy progress bar
            progress = (batch_idx + 1) / train_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(
                f"\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{train_batches} batches",
                end="",
            )
        
        # Progress lr 
        scheduler.step()
        print(f" - Train Loss: {train_epoch_loss/train_batches:.4f}", end="")

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                X, y = batch
                yhat = model(X)
                loss_dict = criterion(yhat, y) 
                weight_dict = criterion.weight_dict
                losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                
                # Calculate loss 
                val_epoch_loss += losses.item() 
                
        print(f" - Test Loss: {val_epoch_loss/test_batches:.4f}")
        
        # Save checkpoints
        if epoch % 25 == 0:
            save(model.state_dict(), f"checkpoints/{epoch+1}_model.pt")
            
    # Final save
    save(model.state_dict(), f"checkpoints/{epoch+1}_model.pt")