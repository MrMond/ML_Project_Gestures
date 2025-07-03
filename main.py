from etc.utils import FPS,RotateTimeseries,pickle_to_tensor as convert_to_tensor
from control_powerpoint import PowerPoint
from model_training import ST_GCN, REVERSE_LOOKUP
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np

# instanciate powerpoint

POWERPOINT_PATH = r"C:\Users\ms216u.MUCLVAD1\OneDrive - Linde Group\Studium\Semester 6\ML_Project\test_.pptx"
assert os.path.isfile(POWERPOINT_PATH), "make sure to enter a correct path to a presentation"
presentation = PowerPoint(POWERPOINT_PATH)

# instanciate the models

CLASSIFICAION_MODEL_PATH = os.path.join(os.getcwd(),"models/classification/classifier_noise0.010.pth")
VISION_MODEL_PATH = os.path.join(os.getcwd(),"models/mediapipe/gesture_recognizer.task")

classification_model = ST_GCN(3)
classification_model.load_state_dict(torch.load(CLASSIFICAION_MODEL_PATH,weights_only=True))
classification_model.eval() # turn off dropout layers

vision_options = vision.GestureRecognizerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path=VISION_MODEL_PATH),
    runnung_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1, # max number of hands detected
)
vision_model = mp.tasks.vision.GestureRecognizer.create_from_options(vision_options)

# evaluate frame

def recognize_hand(frame,timestamp):
    # applychanges to image format for model input
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy().astype(np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # detect hand
    result = vision_model.recognize_for_video(mp_img,timestamp)
    
    # draw landmarks for debug view
    all_landmarks = result.hand_landmarks
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
    except IndexError as e: # if no hand was found
        pass
    # convert to format, that cv2 visualization uses
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) 
    return image,result

def classify_series(tensor,min_confidence=0.65)->str:
    '''returns the class of the tensor or "No class" if no class probability is above the ```min_confidence```'''
    probabilities = F.softmax(classification_model(tensor),dim=1)
    max_probability, idx = torch.max(probabilities,dim=1)
    if max_probability.item() > min_confidence:
        return REVERSE_LOOKUP[idx.item()]
    else:
        return "No class"

# instanciate camera capture

cam = cv2.VideoCapture(0)
FPS_LIMIT=10

# camera loop

skeleton_positions = RotateTimeseries() # assures correct shape for tensor conversion
cooldown = 0

with FPS(limit=FPS_LIMIT) as fps:
    while True:
        if cooldown > 0:
            cooldown -=1

        _,frame = cam.read()

        frame,skeleton = recognize_hand(frame,fps.timestamp_ms)
        skeleton_positions.frames = skeleton

        cv2.imshow("Press 'Q' to exit",frame)

        # check that the data has correct shape for tensor conversion & only do the calculations, if there is no cooldown left
        if skeleton_positions.continuous() and cooldown <= 0: # TODO continuous doesn't work yet
            tensor = convert_to_tensor(skeleton_positions.frames)
            classification = classify_series(tensor)
            match classification:
                case "gesture_backward":
                    presentation.return_slide()
                case "gesture_forward":
                    presentation.advance_slide()
                case "gesture_blacken":
                    presentation.toggle_blacken()
                case _:
                    pass
            cooldown += FPS_LIMIT*3

        # end loop by pressing q
        if cv2.waitKey(1) == ord("q"):
            break
        fps.tick()

cam.release()
cv2.destroyAllWindows()
