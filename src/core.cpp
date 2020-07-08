
#include <vector>

#include "aff/core.h"

#include "opencv2/imgproc.hpp"

#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/opencv.h"

namespace {

bool DetectFace(dlib::frontal_face_detector& detector,
                dlib::shape_predictor& pose_model, cv::Mat3b& srcdst,
                aff::Output& output) {
  dlib::cv_image<dlib::bgr_pixel> cimg(srcdst);

  // Detect faces
  std::vector<dlib::rectangle> org_faces = detector(cimg);

  if (org_faces.empty()) {
    return false;
  }

  unsigned int max_area = 0;
  std::vector<dlib::rectangle> faces(1);
  for (const auto& f : org_faces) {
    if (max_area < f.area()) {
      max_area = f.area();
      faces[0] = f;
    }
  }

  // Find the pose of each face.
  std::vector<dlib::full_object_detection> shapes;
  for (unsigned long i = 0; i < faces.size(); ++i) {
    shapes.push_back(pose_model(cimg, faces[i]));
  }

  output.face_bb =
      cv::Rect(shapes[0].get_rect().left(), shapes[0].get_rect().top(),
               shapes[0].get_rect().width(), shapes[0].get_rect().height());
  output.landmarks.clear();
  for (unsigned int i = 0; i < shapes[0].num_parts(); i++) {
    const auto& part = shapes[0].part(i);
    output.landmarks.push_back(cv::Point2f(part.x(), part.y()));
  }

  return true;
}

bool DrawDetectedFace(const cv::Rect& face_bb,
                      const std::vector<cv::Point2f>& landmarks,
                      cv::Mat3b& vis_landmarks) {
  cv::rectangle(vis_landmarks, face_bb, cv::Scalar(0, 255, 0), 3);

  for (const auto& p : landmarks) {
    cv::circle(vis_landmarks, cv::Point(p.x, p.y), 2, cv::Scalar(0, 0, 255),
               -1);
  }

  return true;
}
}  // namespace

namespace aff {

class AnimeFaceReplacerImpl {
 private:
  dlib::frontal_face_detector detector;
  dlib::shape_predictor pose_model;

 public:
  bool Init(Options options);
  bool Replace(const cv::Mat3b& src, Output& output, const Options& options);
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

bool AnimeFaceReplacerImpl::Replace(const cv::Mat3b& src, Output& output,
                                    const Options& options) {
  cv::Mat3b tmp = src.clone();
  DetectFace(detector, pose_model, tmp, output);
  output.vis_landmarks = src.clone();
  DrawDetectedFace(output.face_bb, output.landmarks, output.vis_landmarks);

  return true;
}

AnimeFaceReplacer::AnimeFaceReplacer() {
  impl = std::make_unique<AnimeFaceReplacerImpl>();
}

AnimeFaceReplacer::~AnimeFaceReplacer() {}

bool AnimeFaceReplacer::Init(Options options) { return impl->Init(options); }

bool AnimeFaceReplacer::Replace(const cv::Mat3b& src, Output& output,
                                const Options& options) {
  return impl->Replace(src, output, options);
}
}  // namespace aff