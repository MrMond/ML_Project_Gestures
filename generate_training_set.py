import os
import add_skeleton

DATA_DIR = r"C:\Users\ms216u.MUCLVAD1\OneDrive - Linde Group\Studium\Semester 6\ML_Project\training\videos"
PKL_DIR = r"C:\Users\ms216u.MUCLVAD1\OneDrive - Linde Group\Studium\Semester 6\ML_Project\training\skeleton_time_series"
CTRL_DIR = r"C:\Users\ms216u.MUCLVAD1\OneDrive - Linde Group\Studium\Semester 6\ML_Project\training\test_vid"

gestures = ["gesture_backward","gesture_blacken","gesture_forward"]

for gesture in gestures:
    for _,_,files in os.walk(os.path.join(DATA_DIR,gesture)):
        for file in files:
            name = file.split(".")[0]
            data_path=os.path.join(DATA_DIR,gesture,file)
            pkl_path = os.path.join(PKL_DIR,gesture,f"{name}.pkl")
            ctrl_path = os.path.join(CTRL_DIR,gesture,f"{name}.mp4")

            add_skeleton.analyse_video(data_path,pkl_path,ctrl_path)
