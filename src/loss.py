import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import sys


def box_cxcywh_to_xyxy(x):
    """Convert bbox from (center_x, center_y, width, height) to (x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert bbox from (x1, y1, x2, y2) to (center_x, center_y, width, height)"""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """Compute area of boxes in xyxy format"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    eps = 1e-7
    iou = inter / (union + eps)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized IoU between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    """
    # Compute regular IoU
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    c_area = wh[:, :, 0] * wh[:, :, 1]
    eps = 1e-7
    return iou - (c_area - union) / (c_area + eps)


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between targets and predictions using 
    the Hungarian algorithm (via scipy's linear_sum_assignment).
    """
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Args:
            cost_class: relative weight of the classification error in the matching cost
            cost_bbox: relative weight of the L1 error of the bounding box coordinates
            cost_giou: relative weight of the giou loss of the bounding box
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching
        
        Params:
            outputs: dict containing:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with class logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with predicted box coordinates
                
            targets: list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] containing class indices
                "boxes": Tensor of dim [num_target_boxes, 4] containing target box coordinates
                
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of selected predictions (in order)
                - index_j is the indices of corresponding selected targets (in order)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and boxes
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            empty = torch.as_tensor([], dtype=torch.int64, device=out_bbox.device)
            return [(empty, empty) for _ in range(bs)]

        tgt_ids = torch.cat([v["labels"] for v in targets]).to(torch.long)  # critical: long dtype
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).to(out_bbox.dtype)
        
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if sizes[i] == 0:
                empty = torch.as_tensor([], dtype=torch.int64, device=C.device)
                indices.append((empty, empty))
            else:
                ii, jj = linear_sum_assignment(c[i])
                indices.append(
                    (torch.as_tensor(ii, dtype=torch.int64), torch.as_tensor(jj, dtype=torch.int64))
                )
        return indices


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) Compute hungarian assignment between ground truth boxes and the outputs of the model
        2) Supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Args:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        
        Args:
            outputs: dict containing:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with class logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with predicted box coordinates
                
            targets: list of dicts, such that len(targets) == batch_size.
                     Each dict contains:
                     "labels": Tensor of dim [num_target_boxes] containing class labels
                     "boxes": Tensor of dim [num_target_boxes, 4] containing box coordinates
        """
        # Check if there are any targets
        if not targets or all(len(t["labels"]) == 0 for t in targets):
            # Return zero losses or handle appropriately
            return {loss: torch.tensor(0.0, device=next(iter(outputs.values())).device) 
                    for loss in self.losses}

        outputs_without_aux = {k: v for k, v in outputs.items()}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        device = next(iter(outputs.values())).device
        # optional: coerce target dtypes defensively
        targets = [
            {'labels': t['labels'].to(torch.long), 'boxes': t['boxes'].to(torch.float32)}
            for t in targets
        ]

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device).clamp(min=1)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




# Example usage
if __name__ == "__main__":
    # Initialize components
    num_classes = 91  # COCO has 80 classes + 1 background
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(num_classes=num_classes-1, matcher=matcher, 
                           weight_dict=weight_dict, eos_coef=0.1, losses=losses)
    
    # Example predictions and targets
    batch_size = 2
    num_queries = 100
    
    # Mock model outputs
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes),
        'pred_boxes': torch.rand(batch_size, num_queries, 4)  # normalized coordinates
    }
    
    # Mock ground truth targets
    targets = [
        {
            'labels': torch.tensor([1, 3, 5]),  # 3 objects of classes 1, 3, 5
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3],   # [cx, cy, w, h] normalized
                                 [0.3, 0.7, 0.1, 0.2],
                                 [0.8, 0.2, 0.15, 0.25]])
        },
        {
            'labels': torch.tensor([2, 7]),  # 2 objects of classes 2, 7
            'boxes': torch.tensor([[0.4, 0.6, 0.3, 0.4],
                                 [0.7, 0.3, 0.2, 0.2]])
        }
    ]
    
    # Compute loss
    loss_dict = criterion(outputs, targets)
    
    # Apply weights to losses
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.4f}")
    print(f"\nTotal weighted loss: {losses.item():.4f}")