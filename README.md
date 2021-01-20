# **aff**: Anime Face Filter
**aff** converts real portlait image/video to anime-like style.
Fully implemented with traditional image processing. No Deep Learning.

Inspired by Anime Face Filter provided by Snapchat and TikTok.



|input|output|
|---|---|
|<img src="https://raw.githubusercontent.com/wiki/unclearness/anime-face-filter/images/lena.jpg" width="180">|<img src="https://raw.githubusercontent.com/wiki/unclearness/anime-face-filter/images/lena_result.png" width="180">|

<img src="https://raw.githubusercontent.com/wiki/unclearness/anime-face-filter/images/trump_result.gif" width="640">


# Dependencies
- OpenCV
  - Image processing and I/O
- dlib
    https://github.com/davisking/dlib
- dlib-models https://github.com/davisking/dlib-models
    - Face and facial landmark detection


# Build
- Install OpenCV 4.x or higher
- `git submodule update --init --recursive`
  - To pull dependencies registered as git submodule. 
- Extract `third_party/dlib-models/shape_predictor_68_face_landmarks.dat.bz2` to the same directory. 
- Use CMake with `CMakeLists.txt`.


# Data
 For video file input test, please download a sample file from [here](https://drive.google.com/file/d/1ovOwdAL7w9WpGF_q_jUdZivpUOcvL7dW/view?usp=sharing) and put `data/test/`


# Executables
- demo.cpp
  - image file input demo
- webcam.cpp
  - webcam stream and video file input demo

# Customization
You can customize images of mouse and eyes. See `data/asset`

# Algorithm
  - Detect face and facial landmarks
  - Reduce and flatten color
  - Replace face parts (eyes and mouse) with anime ones
  - Blend replaced parts into other region


# Problems & Possible Solutions
 - Processing is slow (mainly, face detection and cv::edgePreservingFilter)
   - -> Track face based on the previous frame
   - -> Acceleration (image shrinking, implement faster filter, etc. )
 - Severe jitter
   - -> Consider temporal smoothing
 - Deformation of mouse and eyes is not natural
   - -> Try other deformation methods (polygonize and skinning?)
 - Mouse and eyes are not fit to subject
   - -> Extract and reflect eye and lip color
 - More anime-like style
   - -> Enhance contour lines and create contrasts for them
   - -> ~~CycleGAN~~