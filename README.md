# ML-Project: Gestures

# TODOS

- [x] Capture Video 
- [x] Use Google Model to find hand-rig
- [x] Identify relevant features (see ```training/understand_data.ipynb```)
- [ ] Train gesture classification on Videos
    - [ ] Define gestures
    - [ ] Label data
    - [ ] select model
        - convolutional net _shape: (21 points*3 dimensions | x frames/second * x seconds | n filter?)_
        - normal lin-layer nn
        - table like data $\rarr$ decision trees / random forest?
- [x] Put mediapipe models in stream mode or video mode
- [ ] Stream Video to own model for live gesture 
- [ ] Rig up live gestures to a teams meeting or powerpoint presentation