# ML-Project: Gestures

# TODOS

- [x] Capture Video 
- [x] Use Google Model to find hand-rig
- [x] Identify relevant features (see ```training/understand_data.ipynb```)
- [ ] Train gesture classification on Videos
    - [x] Define gestures
        - record multiple videos of each gesture (~25-50 * team member)
        - apply script to get points and serialize points as pkl-files
    - [x] Label data
    - [x] increase size of dataset via noise injection?
    - [x] select model
    - [x] define model
- [x] Put mediapipe models in stream mode or video mode
- [ ] Stream Video to own model for live gesture 
- [ ] Rig up live gestures to a teams meeting or powerpoint presentation

# Gesture definition:

The starting position for each gesture is a right hand fist, with the fingers pointing towards the camera

## Forward

This will advance the presentation towards the next slide

_Gesture:_ "point the index finger"

1) extend the index finger to the top
2) move the index fnger right and move the hand slightly in that direction


## Backward

This will return the presentation to the previous slide

_Gesture:_ "point the thumb"

1) extend the thumb to the left and move the hand slightly in that direction

## Blacken

This will toggle the screen on and off. 

_Gesture:_ "High five the camera"

1) Open all fingers and move the hand slightly towards the camera
2) hold still for a little while
3) reverse movement ```1.```