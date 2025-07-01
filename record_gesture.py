import os
import cv2
import time
from etc.utils import FPS

TARGET_FRAMERATE = 10 # fps
TARGET_LENGTH = 3 # seconds
CURRENT_GESTURE  = "gesture_forward"

OUT_DIR = os.path.join(os.getcwd(),"training/videos",CURRENT_GESTURE)
assert os.path.isdir(OUT_DIR)

def record_gesture(fps:int,length:int,out_path:str,cam:cv2.VideoCapture):
    '''records the main camera of the device and assures that the resulting mp4 has exactly ```length*fps``` frames'''
    frame_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        filename=out_path,
        fourcc=fourcc,
        fps=fps,
        frameSize=(frame_w, frame_h),
    )

    frame_count = fps*length

    with FPS(limit=fps) as f:
        while frame_count>0:
            _, frame = cam.read()
            cv2.imshow("Recording",frame)
            out.write(frame)
            frame_count-=1
            f.tick()
        
    out.release()
    cv2.destroyWindow("Recording")

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    try: # don't override existing videos
        i = max([int(i.split(".")[0]) for i in os.listdir(OUT_DIR)]) + 1
    except:
        i = 0
    while True:
        _, frame = cam.read()
        cv2.imshow("Cam",frame)
        match cv2.waitKey(1):
            case 32: # ord(" ")
                record_gesture(TARGET_FRAMERATE,TARGET_LENGTH,os.path.join(OUT_DIR,f"{i:03}.mp4"),cam)
                i+=1
            case 113:# ord("q")
                break
    cam.release()
    cv2.destroyAllWindows()