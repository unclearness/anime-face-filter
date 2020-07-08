
#include "aff/core.h"

#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"

namespace aff {

class AnimeFaceReplacerImpl {
 private:
  dlib::frontal_face_detector detector;
  dlib::shape_predictor pose_model;
 public:
  bool Init(Options options);
  bool Replace(const cv::Mat3b& src, cv::Mat3b& dst, const Options& options);
};

bool AnimeFaceReplacerImpl::Init(Options options) {
  detector = dlib::get_frontal_face_detector();
  try {
    dlib::deserialize(options.dlib_model_path) >> pose_model;
  } catch (dlib::serialization_error e) {
    return false;
  }
  return true;
}

bool AnimeFaceReplacerImpl::Replace(const cv::Mat3b& src, cv::Mat3b& dst,
                                  const Options& options) {

  return true;
}


AnimeFaceReplacer::AnimeFaceReplacer() {
  impl = std::make_unique<AnimeFaceReplacerImpl>();
}
AnimeFaceReplacer::~AnimeFaceReplacer() {}

bool AnimeFaceReplacer::Init(Options options) { return impl->Init(options); }

bool AnimeFaceReplacer::Replace(const cv::Mat3b& src, cv::Mat3b& dst,
                                const Options& options) {
  return impl->Replace(src, dst, options);
}
}  // namespace aff