from data import DETRData
from model import DETR
import torch
from torch import load
from torch.utils.data import DataLoader 
from matplotlib import pyplot as plt 
from utils.boxes import rescale_bboxes
from utils.setup import get_classes

num_classes = 3
test_dataset = DETRData('data/test', train=False) 
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4, drop_last=True) 
model = DETR(num_classes=num_classes)
model.eval()
model.load_state_dict(load('pretrained/4426_model.pt'))

X, y = next(iter(test_dataloader))

result = model(X) 
# print(result) 
# print(result['pred_logits'].shape) 
# print(result['pred_boxes'].shape) 

probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
max_probs, max_classes = probabilities.max(-1)
keep_mask = max_probs > 0.95
batch_indices, query_indices = torch.where(keep_mask) 

bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (224,224))
classes = max_classes[batch_indices, query_indices]
probas = max_probs[batch_indices, query_indices]
print(batch_indices, query_indices)
print(classes)
print(probas)
print(bboxes) 

CLASSES = get_classes()

fig, ax = plt.subplots(2,2) 
axs = ax.flatten()
for idx, (img, ax) in enumerate(zip(X, axs)): 
    ax.imshow(img.permute(1,2,0))
    for batch_idx, box_class, box_prob, bbox in zip(batch_indices, classes, probas, bboxes): 
        if batch_idx == idx: 
            xmin, ymin, xmax, ymax = bbox.detach().numpy()
            print(xmin, ymin, xmax, ymax) 
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
            text = f'{CLASSES[box_class]}: {box_prob:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

fig.tight_layout() 
plt.show()     