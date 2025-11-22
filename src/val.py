import torch
from torch import load
from torch.utils.data import DataLoader 
from tqdm import tqdm
import time

# --- Import torchmetrics ---
from torchmetrics.detection import MeanAveragePrecision

from data import DETRData
from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes
from utils.logger import get_logger

# --- Setup ---
logger = get_logger("test")
logger.print_banner()

# --- Config ---
num_classes = 128
BATCH_SIZE = 4 
CHECKPOINT_PATH = 'checkpoints/999_model.pt'
device = torch.device('cpu') # Run on CPU
logger.info(f"Running evaluation on device: {device}")

# --- Load Data ---
# shuffle=False is important for evaluation!
test_dataset = DETRData('combineddata/val', train=False) 
test_dataloader = DataLoader(test_dataset, 
                             shuffle=False, 
                             batch_size=BATCH_SIZE, 
                             drop_last=False) # Process all images
CLASSES = get_classes()

# --- Load Model ---
model = DETR(num_classes=num_classes)
try:
    model.load_state_dict(load(CHECKPOINT_PATH, map_location=device))
    logger.info(f"Successfully loaded model weights from {CHECKPOINT_PATH}")
except Exception as e:
    logger.warning(f"Direct load failed ({e}). Trying to load from checkpoint dict...")
    checkpoint = load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Successfully loaded model weights from checkpoint dict.")
model.to(device)
model.eval()

# --- Instantiate mAP Metric ---
# This object will accumulate results from all batches
# We set class_metrics=True to see mAP for each Vietnamese class
map_metric = MeanAveragePrecision(class_metrics=True)
map_metric.to(device)

logger.test("Starting mAP evaluation on validation set...")

# --- Main Evaluation Loop ---
for X, y in tqdm(test_dataloader, desc="Calculating mAP"):
    
    X = X.to(device)
    current_batch_size = X.shape[0]
    H, W = X.shape[2:]

    # --- Run Inference ---
    with torch.no_grad():
        result = model(X) 

    # --- Format Predictions (for torchmetrics) ---
    logits = result['pred_logits'].to(device)
    boxes_norm = result['pred_boxes'].to(device) # Normalized cxcywh
    
    preds = []
    for i in range(current_batch_size): 
            scores_all = logits[i].softmax(-1)
            scores, labels = scores_all[:, :-1].max(-1) 
            boxes_abs = rescale_bboxes(boxes_norm[i], (H, W)) 
            
            preds.append({
                'boxes': boxes_abs,
                'scores': scores,
                'labels': labels
            })

    # --- Format Targets (Ground Truth) (for torchmetrics) ---
    # We must also convert your ground-truth 'y' to absolute xyxy
    targets = []
    for i in range(current_batch_size):
        # 'y' is a list, so y[i] is the dict for the i-th image
        # We assume 'y' boxes are also normalized cxcywh
        boxes_norm_gt = y['boxes'][i].to(device) 
        labels_gt = y['labels'][i].to(device)
        
        # Un-normalize boxes to absolute xyxy format
        boxes_abs_gt = rescale_bboxes(boxes_norm_gt, (H, W))
        
        targets.append({
            'boxes': boxes_abs_gt,
            'labels': labels_gt
        })

    # --- Update metric with batch results ---
    map_metric.update(preds, targets)

# --- Compute and Print Final Results ---
logger.info("--- Evaluation Complete ---")
final_map_scores = map_metric.compute()

# Print the main mAP scores
logger.info(f"mAP (all):               {final_map_scores['map']:.4f}")
logger.info(f"mAP (small):             {final_map_scores['map_small']:.4f}")
logger.info(f"mAP (medium):            {final_map_scores['map_medium']:.4f}")
logger.info(f"mAP (large):             {final_map_scores['map_large']:.4f}")
logger.info(f"mAP @ .50 IoU:           {final_map_scores['map_50']:.4f}")
logger.info(f"mAP @ .75 IoU:           {final_map_scores['map_75']:.4f}")

# Print mAP for each class (including Vietnamese)
logger.info("\n--- Per-Class mAP ---")
for i, (map_val, mar_val) in enumerate(zip(final_map_scores['map_per_class'], final_map_scores['mar_100_per_class'])):
    if i < len(CLASSES): # Ensure we don't go out of bounds
        class_name = CLASSES[i]
        logger.info(f"[{class_name}]: mAP = {map_val:.4f}, mAR = {mar_val:.4f}")