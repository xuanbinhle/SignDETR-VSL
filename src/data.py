import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import os 
from PIL import Image 
import albumentations as A
import numpy as np
from colorama import Fore 
from matplotlib import pyplot as plt 
from utils.boxes import rescale_bboxes, stacker
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import DataLoaderHandler
import sys 
import urllib.parse

class DETRData(Dataset): 
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, 'labels')
        self.images_path = os.path.join(self.path, 'images')
        self.label_files = os.listdir(self.labels_path) 
        self.labels = list(filter(lambda x: x.endswith('.txt'), self.label_files))
        self.train = train
        
        # Initialize logger
        self.logger = get_logger("data_loader")
        self.data_handler = DataLoaderHandler()
        
        # Log dataset initialization
        dataset_info = {
            "Dataset Path": self.path,
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": self.images_path,
            "Labels Path": self.labels_path
        }
        self.data_handler.log_dataset_stats(dataset_info)
        
        # Log transforms information
        transform_list = [
            "Resize to 500x500",
            "Random Crop 224x224 (training only)",
            "Final Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Color Jitter (training only)",
            "Normalize (ImageNet stats)",
            "Convert to Tensor"
        ]
        self.data_handler.log_transform_info(transform_list)           

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        self.transform = A.Compose(
            [   
                A.Resize(500,500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if self.train else []), # Example random crop
                A.Resize(224,224),
                *([A.HorizontalFlip(p=0.5)] if self.train else []),
                *([A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5)] if self.train else []),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']) # This defines the key
        )
        
        for attempt in range(max_attempts):
            try:
                # We pass the labels using the key 'class_labels'
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
                # Check if we still have bboxes after transformation
                if len(transformed['bboxes']) > 0:
                    return transformed
            except:
                continue
        
        # Fallback in case augmentation fails, returning un-augmented data
        # To avoid this, we'll re-run the transform without the cropping that might remove boxes
        # This part is a bit risky, but better than returning nothing
        try:
            self.transform_safe = A.Compose(
                [   
                    A.Resize(224,224), # No random crop
                    *([A.HorizontalFlip(p=0.5)] if self.train else []),
                    *([A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5)] if self.train else []),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    A.ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
            )
            transformed = self.transform_safe(image=image, bboxes=bboxes, class_labels=labels)
            return transformed
        except Exception as e:
            self.logger.error(f"Safe transform fallback failed: {e}")
            # As a last resort, return the original data transformed to tensor
            # This might cause errors if bounding boxes are empty, but it's better than KeyError
            self.transform_last_resort = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            return self.transform_last_resort(image=image, bboxes=bboxes, class_labels=labels)


    def __len__(self): 
        return len(self.labels) 

    # def __getitem__(self, idx): 
    #     self.label_path = os.path.join(self.labels_path, self.labels[idx]) 
    #     self.image_name = self.labels[idx].split('.')[0]
    #     self.image_path = os.path.join(self.images_path, f'{self.image_name}.jpg') 
        
    #     img = Image.open(self.image_path)
    #     with open(self.label_path, 'r') as f: 
    #         annotations = f.readlines()
    #     class_labels = []
    #     bounding_boxes = []
    #     for annotation in annotations: 
    #         annotation = annotation.split('\n')[:-1][0].split(' ')
    #         class_labels.append(annotation[0]) 
    #         bounding_boxes.append(annotation[1:])
    #     class_labels = np.array(class_labels).astype(int) 
    #     bounding_boxes = np.array(bounding_boxes).astype(float) 

    #     augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
    #     augmented_img_tensor = augmented['image']
    #     augmented_bounding_boxes = np.array(augmented['bboxes'])
    #     augmented_classes = augmented['class_labels'] # This line was correct in your commented-out code

    #     labels = torch.tensor(augmented_classes, dtype=torch.long)   
    #     boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)
    #     return augmented_img_tensor, {'labels': labels, 'boxes': boxes}

    def __getitem__(self, idx): 
        try:
            #Decode URL-encoded filenames ---
            # Get the raw filename (e.g., "...L%C6%B0%E1%BB%A3c.txt")
            raw_label_filename = self.labels[idx]
            
            # Decode it to a normal path (e.g., "...Lược.txt")
            decoded_label_filename = urllib.parse.unquote(raw_label_filename)

            #Use local variables, not 'self.' ---
            # Building paths using the DECODED name
            label_path = os.path.join(self.labels_path, decoded_label_filename) 
            
            # Use os.path.splitext for a safer way to get the name
            image_name = os.path.splitext(decoded_label_filename)[0]
            
            # --- Robust image path finding ---
            # Check for multiple common extensions
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                potential_path = os.path.join(self.images_path, f'{image_name}{ext}')
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                self.logger.error(f"Image file not found for label: {decoded_label_filename}")
                self.logger.error(f"Looked for: {image_name} with extensions .jpg, .png, etc.")
                # Return None or raise error. Returning None will be caught by collate_fn
                return None # This will be skipped by a default collate_fn, but your 'stacker' might need to handle it

            # These 'open' calls will now work
            # Use .convert('RGB') to handle grayscale or RGBA images
            img = Image.open(image_path).convert('RGB')
            with open(label_path, 'r') as f: 
                annotations = f.readlines()

            class_labels = []
            bounding_boxes = []
            for annotation in annotations: 
                # More robust line parsing ---
                # .strip() removes whitespace (like '\n') from start/end
                parts = annotation.strip().split(' ') 
                if not parts or len(parts) < 5: # Skip empty or invalid lines
                    continue
                    
                class_labels.append(parts[0]) 
                bounding_boxes.append(parts[1:])
            
            # --- Check for empty labels ---
            if not class_labels:
                self.logger.warning(f"No valid annotations found in {label_path}. Using a dummy box.")
                # Albumentations can fail with empty bboxes. 
                # We can skip this image or provide a dummy box. Let's try skipping.
                # A better approach might be to provide a dummy box off-screen if skipping is hard.
                # For now, let's keep it but be aware. If this fails, we must provide a dummy.
                # Let's provide a dummy box [0, 0, 0, 0] for class 0
                if not bounding_boxes:
                    class_labels = [0] # dummy class
                    bounding_boxes = [[0.0, 0.0, 0.0, 0.0]] # dummy box
            
            class_labels = np.array(class_labels).astype(int) 
            bounding_boxes = np.array(bounding_boxes).astype(float) 

            augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
            augmented_img_tensor = augmented['image']
            augmented_bounding_boxes = np.array(augmented['bboxes'])
            
            # Albumentations output key matches input key ---
            # --- THIS IS THE FIX ---
            # The key is 'class_labels' because of line 89
            augmented_classes = augmented['class_labels']

            labels = torch.tensor(augmented_classes, dtype=torch.long)   
            boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)
            
            return augmented_img_tensor, {'labels': labels, 'boxes': boxes}
        
        except Exception as e:
            self.logger.error(f"Error loading data at index {idx} (file: {self.labels[idx]}): {e}")
            # We must return something. Let's return the next item to avoid crashing the batch.
            # A better way is to handle 'None' in the collate_fn
            return self.__getitem__((idx + 1) % len(self)) # Try loading the next sample


if __name__ == '__main__':
    # --- IMPORTANT: Update this path to point to your *combineddata* folder ---
    dataset = DETRData('D:\SignDETR-VSL\combineddata', train=True) 
    
    # Add a custom collate_fn to filter out None values
    def custom_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None, None # Handle empty batch
        return stacker(batch)

    dataloader = DataLoader(dataset, collate_fn=custom_collate, batch_size=4, drop_last=True)

    try:
        X, y = next(iter(dataloader))
        
        if X is None:
            print(Fore.RED + "Error: Dataloader returned an empty batch. Check image paths and label files." + Fore.RESET)
            sys.exit()

        print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET) 
        
        # --- IMPORTANT: Update this path to get your *combined* classes ---
        # This will fail unless you have a get_classes() that reads your *new* data.yaml
        # For now, let's create a dummy list based on your train.py
        # CLASSES = get_classes() 
        CLASSES = [str(i) for i in range(128)]  
        
        fig, ax = plt.subplots(2,2) 
        axs = ax.flatten()
        for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)): 
            # De-normalize image for plotting
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display * std + mean).clip(0, 1) # De-normalize
            
            ax.imshow(img_display)
            box_classes = annotations['labels'] 
            boxes = rescale_bboxes(annotations['boxes'], (224,224))
            for box_class, bbox in zip(box_classes, boxes): 
                if box_class < len(CLASSES): # Safety check
                    xmin, ymin, xmax, ymax = bbox.detach().numpy()
                    # print(xmin, ymin, xmax, ymax) 
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
                    text = f'{CLASSES[box_class]}'
                    ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
                else:
                    print(f"Warning: Class ID {box_class} is out of range for plotting.")

        fig.tight_layout() 
        plt.show()
    
    except StopIteration:
        print(Fore.RED + "Dataloader is empty. This could be due to:" + Fore.RESET)
        print("1. The 'combineddata' directory is empty or not found.")
        print("2. No image-label pairs were successfully matched by dataset_combiner.py.")
        print("3. All samples failed to load in __getitem__.")
    except Exception as e:
        print(f"An error occurred during dataloader test: {e}")