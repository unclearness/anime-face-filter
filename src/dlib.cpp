
#include <vector>

#include "src/dlib.h"

#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/opencv.h"

namespace {

bool DetectFace(dlib::frontal_face_detector& detector,
                dlib::shape_predictor& pose_model, cv::Mat3b& srcdst,
                cv::Rect& face_bb, std::vector<cv::Point>& landmarks) {
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

  face_bb =
      cv::Rect(shapes[0].get_rect().left(), shapes[0].get_rect().top(),
               shapes[0].get_rect().width(), shapes[0].get_rect().height());
  landmarks.clear();
  for (unsigned int i = 0; i < shapes[0].num_parts(); i++) {
    const auto& part = shapes[0].part(i);
    landmarks.push_back(cv::Point2f(part.x(), part.y()));
  }

  return true;
}

}  // namespace

namespace aff {

class DlibFaceDetectorImpl {
 private:
  dlib::frontal_face_detector detector;
  dlib::shape_predictor pose_model;

 public:
  bool Init(const std::string& dlib_model_path);
  bool Detect(cv::Mat3b& srcdst, cv::Rect& face_bb,
              std::vector<cv::Point>& landmarks);
};

bool DlibFaceDetectorImpl::Init(const std::string& dlib_model_path) {
  detector = dlib::get_frontal_face_detector();
  try {
    dlib::deserialize(dlib_model_path) >> pose_model;
  } catch (dlib::serialization_error e) {
    return false;
  }

  return true;
}

bool DlibFaceDetectorImpl::Detect(cv::Mat3b& srcdst, cv::Rect& face_bb,
                                  std::vector<cv::Point>& landmarks) {
  return DetectFace(detector, pose_model, srcdst, face_bb, landmarks);
}

DlibFaceDetector::DlibFaceDetector() {
  impl = std::make_unique<DlibFaceDetectorImpl>();
}

DlibFaceDetector::~DlibFaceDetector() {}

bool DlibFaceDetector::Init(const std::string& dlib_model_path) {
  return impl->Init(dlib_model_path);
}

bool DlibFaceDetector::Detect(cv::Mat3b& srcdst, cv::Rect& face_bb,
                              std::vector<cv::Point>& landmarks) {
  return impl->Detect(srcdst, face_bb, landmarks);
}

}  // namespace aff