import torch 

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def stacker(batch):
    """
    Custom collate function for DETR.
    
    Handles batches with varying numbers of objects per image.
    The images get stacked, but targets remain as a list since
    each image can have a different number of objects.
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack all images into a single tensor
    # This works because all images have the same size after transforms
    images = torch.stack(images, dim=0)
    
    # Keep targets as a list - each element corresponds to one image
    # This allows each image to have a different number of objects
    return images, targets