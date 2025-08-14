# Example of how to modify your training loop in train.py
# Replace lines 60-98 with this pattern:

# Create progress bar for current epoch
with training_handler.create_training_progress() as epoch_progress:
    # Use the new method to add task with initial loss values
    epoch_task = training_handler.add_task_with_losses(
        epoch_progress, 
        f"[green]Epoch {epoch+1}/{epochs}", 
        total=train_batches
    )
    
    for batch_idx, batch in enumerate(train_dataloader): 
        X, y = batch
        try: 
            yhat = model(X) 
            yhat_classes = yhat['pred_logits'] 
            yhat_bb = yhat['pred_boxes'] 
            loss_dict = criterion(yhat, y) 
            weight_dict = criterion.weight_dict
            
            # Calculate individual loss components
            ce_loss = loss_dict['labels']['loss_ce'].item()
            bbox_loss = loss_dict['boxes']['loss_bbox'].item()
            giou_loss = loss_dict['boxes']['loss_giou'].item()
            
            # Ensure we sum exactly over the expected weighted keys, and keep tensor dtype
            losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
            total_loss = losses.item()
            
            # Calculate loss 
            train_epoch_loss += total_loss
            
            # Zero grads
            opt.zero_grad()
            
            # Backward
            losses.backward()
            # Apply
            opt.step()
            
            # Update progress with real-time loss display (instead of the old periodic logging)
            training_handler.update_progress_with_losses(
                epoch_progress, 
                epoch_task, 
                advance=1,
                total_loss=total_loss,
                ce_loss=ce_loss,
                bbox_loss=bbox_loss,
                giou_loss=giou_loss
            )
            
            # Remove the old periodic logging:
            # if batch_idx % 10 == 0:
            #     loss_components = {...}
            #     training_handler.log_loss_components(...)
            
        except Exception as e: 
            logger.error(f"Training error at epoch {epoch}, batch {batch_idx}: {str(e)}")
            logger.error(f"Batch targets: {str(y)}")
            sys.exit() 