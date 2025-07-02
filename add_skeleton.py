import os
import cv2
import pickle
import numpy as np
# models
# https://ai.google.dev/edge/mediapipe/solutions/guide
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

MODEL_PATH = os.path.join(os.getcwd(), "models", "mediapipe", "gesture_recognizer.task")
VID_OPTIONS = vision.GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,  # max nuber of hands detected
)
VID_RECOGNIZER = mp.tasks.vision.GestureRecognizer.create_from_options(VID_OPTIONS)

IMG_OPTIONS = vision.GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2, # max number of hands detected
)
IMG_RECOGNIZER = mp.tasks.vision.GestureRecognizer.create_from_options(IMG_OPTIONS)

def add_skeleton(frame, frame_timestamp_ms):
    frame.astype(np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result = VID_RECOGNIZER.recognize_for_video(mp_img, frame_timestamp_ms)

    hand = result.handedness  # left or right hand
    all_landmarks = result.hand_landmarks  # 21 points as a hand rig

    image = mp_img.numpy_view().copy()
    try:
        for hand_landmarks in all_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
    except IndexError as e:
        pass

    return image, result

def add_skeleton_per_image(frame):
    frame.astype(np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result = IMG_RECOGNIZER.recognize(mp_img)

    all_landmarks = result.hand_landmarks  # 21 points as a hand rig

    image = mp_img.numpy_view().copy()
    try:
        for hand_landmarks in all_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
    except IndexError as e:
        pass

    return image, result

def analyse_video(video_path:str,pickle_path:str,control_path:str):
    # reader
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        filename=control_path,
        fourcc=fourcc,
        fps=fps,
        frameSize=(frame_w, frame_h),
    )

    frame_results = {}

    while True: # analyze video
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        timestamp = frame_idx/fps
        frame_idx += 1

        frame, result = add_skeleton_per_image(frame)#,int(timestamp*100))

        if result:
            frame_results[timestamp] = result
            out.write(frame)
    
    with open(pickle_path,"wb") as of: # generate output
        pickle.dump(frame_results,of)

    out.release()
    cap.release()



if __name__ == "__main__":

    # redundant code:

    import pickle, cv2

    data_dir =  r"training\test_vid"

    for vid in os.listdir(data_dir):
        if not vid.split(".")[-1] == "mp4":
            continue
        cap = cv2.VideoCapture(os.path.join(data_dir,vid))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        frame_results = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            frame_idx += 1

            frame,result = add_skeleton(frame,int(timestamp*100))

            if result:
                frame_results[timestamp] = result

        with open(os.path.join(data_dir,f"{vid.split(".")[0]}.pkl"),"wb") as of:
            pickle.dump(frame_results,of)
        cap.release()