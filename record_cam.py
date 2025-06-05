import cv2
import uuid
import os
from etc.utils import FPS

# relative imports
from add_skeleton import add_skeleton

record_duration = 10  # s
target_framerate = 24  # fps
instructions = "Press 'Q' to exit"
store_recording = False

# define camera

cam = cv2.VideoCapture(0)
frame_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
if store_recording:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        filename=os.path.join(os.getcwd(), "training", "test_vid", f"{uuid.uuid4()}.mp4"),
        fourcc=fourcc,
        fps=target_framerate,
        frameSize=(frame_w, frame_h),
    )

with FPS(limit=target_framerate) as fps:
    while True:
        ret, frame = cam.read()

        frame = add_skeleton(frame,fps.timestamp_ms)

        if store_recording:
            out.write(frame)

        cv2.putText(
            frame,
            f"{instructions} ({fps.measured_fps} fps)",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Cam", frame)

        if cv2.waitKey(1) == ord("q"):
            break
        
        fps.tick()

cam.release()
if store_recording:
    out.release()
cv2.destroyAllWindows()
