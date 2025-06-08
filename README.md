# ML-Project: Gestures

# TODOS

- [x] Capture Video 
- [x] Use Google Model to find hand-rig
- [x] Identify relevant features (see ```training/understand_data.ipynb```)
- [ ] Train gesture classification on Videos
    - [ ] Define gestures
        - record multiple videos of each gesture (~25-50 * team member)
        - apply script to get points and serialize points as pkl-files
    - [ ] Label data
        - increase size of dataset via noise injection?
    - [ ] select model
        - convolutional net _shape: (21 points*3 dimensions | x frames/second * x seconds | n filter?)_
        - normal lin-layer nn
        - flatten list of coordinates and the apply 1D convolution with stride = 3?
        - table like data $\rarr$ decision trees / random forest?
- [x] Put mediapipe models in stream mode or video mode
- [ ] Stream Video to own model for live gesture 
- [ ] Rig up live gestures to a teams meeting or powerpoint presentation
