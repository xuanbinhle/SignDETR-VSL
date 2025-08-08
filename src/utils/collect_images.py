import cv2
import numpy as np
import uuid
import time 
import os 
from colorama import Fore 

classes = {
    0:'hello', 
    1:'iloveyou', 
    2:'thankyou'
}

class CaptureImages(): 
    def __init__(self, path:str, classes:dict, camera_id:int) -> None: 
        self.cap = cv2.VideoCapture(camera_id) 
        self.path = path 
        self.classes = classes 

    def capture(self, class_name:str) -> str:     
        try: 
            ret, frame = self.cap.read() 
            image = cv2.putText(frame, f'Capturing {class_name}', (0,100), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Image Capture', image)
            cv2.imwrite(os.path.join(self.path, f'{class_name}-{uuid.uuid1()}.jpg'), frame) 
            if cv2.waitKey(1) & 0xFF==ord('q'):
                pass
        except Exception as e: 
            return Fore.LIGHTRED_EX + f'Capture failure: {e}' + Fore.RESET
        return Fore.LIGHTGREEN_EX + 'Capture success' + Fore.RESET

    def run(self, sleep_time:int = 1, num_images=10):
        for img_class in self.classes.values(): 
            for idx in range(num_images): 
                print(self.capture(img_class)) 
                time.sleep(sleep_time) 

if __name__ == '__main__': 
    cap = CaptureImages('./data/test', classes, 0) 
    cap.run(num_images=30)
