"""This script was necessairy to improve contrast of the training videos (they are not included in the repo),\
   as some of them had too little detail for the hand-tracking-model. It is not used in the training process"""

import cv2
import os

DIR = r"C:\Users\ms216u.MUCLVAD1\OneDrive - Linde Group\Studium\Semester 6\ML_Project\training\videos"


def darken_vid(path: str, outpath: str):
    # reader
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        filename=outpath,
        fourcc=fourcc,
        fps=fps,
        frameSize=(frame_w, frame_h),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.convertScaleAbs(
            frame, alpha=1.5, beta=-50
        )  # darken and add contrast
        out.write(frame)
    cap.release()
    out.release()


for root, dirs, files in os.walk(DIR):
    for file in files:
        path = os.path.join(root, file)
        outpath = os.path.join(root, f"n{file}")
        darken_vid(path, outpath)
