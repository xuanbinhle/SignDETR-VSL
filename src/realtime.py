import cv2
import torch
from torch import load
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
import sys 


transforms = A.Compose(
        [   
            A.Resize(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
    )

model = DETR(num_classes=3)
model.eval()
model.load_state_dict(load('pretrained/4426_model.pt'))
CLASSES = get_classes() 
COLORS = get_colors() 
cap = cv2.VideoCapture(0)

while cap.isOpened(): 
    ret, frame = cap.read()
    transformed = transforms(image=frame)
    result = model(torch.unsqueeze(transformed['image'], dim=0))

    probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > 0.8

    batch_indices, query_indices = torch.where(keep_mask) 

    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (1920,1080))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    print(bboxes, classes, probas) 

    for bclass, bprob, bbox in zip(classes, probas, bboxes): 
        bclass = bclass.detach().numpy()
        bprob = bprob.detach().numpy() 
        x1,y1,x2,y2 = bbox.detach().numpy()
        frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), COLORS[bclass], 10)
        frame_text = f"{CLASSES[bclass]} - {round(float(bprob),4)}"
        frame = cv2.rectangle(frame, (int(x1),int(y1)-100), (int(x1)+700,int(y1)), COLORS[bclass], -1)
        frame = cv2.putText(frame, frame_text, (int(x1),int(y1)), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 4, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows() 
