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
- demo.ccpp
  - image file input demo
- webcam.cpp
  - webcam stream and video file input demo

# Customization
You can customize images of mouse and eyes. See `data/asset`


 # Notes
 - Current implemtation is slow.
   - face detection
   - cv::edgePreservingFilter
 - Severe jitter since temporal smoothing is not cosidered yet.

