#pragma once

#include <memory>

#include "opencv2/core.hpp"

#define AFF_DLIB_DEFAULT_MODEL_PATH \
  "../third_party/dlib-models/shape_predictor_68_face_landmarks.dat"

namespace aff {

struct Options {
  std::string dlib_model_path = AFF_DLIB_DEFAULT_MODEL_PATH;
  std::string debug_dir = "";
};

class AnimeFaceReplacerImpl;

class AnimeFaceReplacer {
 private:
  std::unique_ptr<AnimeFaceReplacerImpl> impl;

 public:
  AnimeFaceReplacer();
  ~AnimeFaceReplacer();
  bool Init(Options options);
  bool Replace(const cv::Mat3b& src, cv::Mat3b& dst, const Options& options);
};

}  // namespace aff