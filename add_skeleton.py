import os
import numpy as np
# models
# https://ai.google.dev/edge/mediapipe/solutions/guide
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

model_path = os.path.join(os.getcwd(), "models", "mediapipe", "gesture_recognizer.task")
OPTIONS = vision.GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,  # max nuber of hands detected
)
RECOGNIZER = mp.tasks.vision.GestureRecognizer.create_from_options(OPTIONS)


def add_skeleton(frame, frame_timestamp_ms):
    frame.astype(np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result = RECOGNIZER.recognize_for_video(mp_img, frame_timestamp_ms)

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

    # todo: return frame as np
    return image, result

if __name__ == "__main__":

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