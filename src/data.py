import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import os 
from PIL import Image 
import albumentations as A
import numpy as np
from colorama import Fore 
from matplotlib import pyplot as plt 
from utils.boxes import box_cxcywh_to_xyxy, rescale_bboxes, stacker


class DETRData(Dataset): 
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, 'labels')
        self.images_path = os.path.join(self.path, 'images')
        self.label_files = os.listdir(self.labels_path) 
        self.labels = list(filter(lambda x: x.endswith('.txt'), self.label_files))
        self.transform = A.Compose(
            [   
                A.Resize(500,500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if train else []), # Example random crop
                A.Resize(224,224),
                *([A.HorizontalFlip(p=0.5)] if train else []),
                *([A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5)] if train else []),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    def __len__(self): 
        return len(self.labels) 

    def __getitem__(self, idx): 
        self.label_path = os.path.join(self.labels_path, self.labels[idx]) 
        self.image_name = self.labels[idx].split('.')[0]
        self.image_path = os.path.join(self.images_path, f'{self.image_name}.jpg') 
        
        # Handle image 
        img = Image.open(self.image_path)
        
        # print(img_tensor.shape) 
        # Handle annotations
        with open(self.label_path, 'r') as f: 
            annotations = f.readlines()
        class_labels = []
        bounding_boxes = []
        for annotation in annotations: 
            annotation = annotation.split('\n')[:-1][0].split(' ')
            class_labels.append(annotation[0]) 
            bounding_boxes.append(annotation[1:])
        class_labels = np.array(class_labels).astype(int) 
        bounding_boxes = np.array(bounding_boxes).astype(float) 

        augmented = self.transform(image=np.array(img), bboxes=bounding_boxes, class_labels=class_labels)
        augmented_img_tensor = augmented['image']
        augmented_bounding_boxes = augmented['bboxes']
        augmented_classes = augmented['class_labels']
        # if augmented_bounding_boxes.shape[0] == 0: 
        #     augmented_bounding_boxes = np.zeros((1,4))
        #     augmented_classes = [3]

        augmented_bounding_boxes = np.array(augmented['bboxes']) 

        # return img_tensor, {'labels':torch.tensor(class_labels), 'boxes': torch.tensor(bounding_boxes, dtype=torch.float32)}
        labels = torch.tensor(augmented_classes, dtype=torch.long)          # int64 even when empty
        boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32) # (N,4) or (0,4)
        return augmented_img_tensor, {'labels': labels, 'boxes': boxes}

if __name__ == '__main__':
    dataset = DETRData('data/train', train=True) 
    # rand = np.random.randint(dataset.__len__()) 
    # X,y = dataset.__getitem__(rand)

    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)
    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET) 
    # CLASSES = ['hello', 'iloveyou', 'thankyou', 'noclass']     
    # fig, ax = plt.subplots(2,2) 
    # axs = ax.flatten()
    # for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)): 
    #     ax.imshow(img.permute(1,2,0))
    #     box_classes = annotations['labels'] 
    #     boxes = rescale_bboxes(annotations['boxes'], (224,224))
    #     for box_class, bbox in zip(box_classes, boxes): 
    #         if box_class != 3: 
    #             xmin, ymin, xmax, ymax = bbox.detach().numpy()
    #             print(xmin, ymin, xmax, ymax) 
    #             ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
    #             text = f'{CLASSES[box_class]}'
    #             ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    # fig.tight_layout() 
    # plt.show()     