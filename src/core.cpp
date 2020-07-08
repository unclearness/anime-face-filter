
#include <vector>

#include "aff/core.h"

#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/opencv.h"

namespace {

int DLIB_CHIN[17] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
int DLIB_R_EYEBLOW[5] = {17, 18, 19, 20, 21};
int DLIB_L_EYEBLOW[5] = {22, 23, 24, 25, 26};
int DLIB_OUTER[27] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
int DLIB_NOSE_UPPER[4] = {27, 28, 29, 30};
int DLIB_NOSE_LOWER[5] = {31, 32, 33, 34, 35};
int DLIB_R_EYE[6] = {36, 37, 38, 39, 40, 41};
int DLIB_L_EYE[6] = {42, 43, 44, 45, 46, 47};
int DLIB_MOUSE_OUTER_NUM = 12;
int DLIB_MOUSE_OUTER[12] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
int DLIB_MOUSE_INNER[8] = {60, 61, 62, 63, 64, 65, 66, 67};

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
                      const std::vector<cv::Point>& landmarks,
                      cv::Mat3b& vis_landmarks) {
  cv::rectangle(vis_landmarks, face_bb, cv::Scalar(0, 255, 0), 3);

  for (const auto& p : landmarks) {
    cv::circle(vis_landmarks, cv::Point(p.x, p.y), 2, cv::Scalar(0, 0, 255),
               -1);
  }

  return true;
}

// 画像を画像に貼り付ける関数
void paste(cv::Mat dst, cv::Mat src, int x, int y, int width, int height) {
  cv::Mat resized_img;
  cv::resize(src, resized_img, cv::Size(width, height));

  if (x >= dst.cols || y >= dst.rows) return;
  int w = (x >= 0) ? std::min(dst.cols - x, resized_img.cols)
                   : std::min(std::max(resized_img.cols + x, 0), dst.cols);
  int h = (y >= 0) ? std::min(dst.rows - y, resized_img.rows)
                   : std::min(std::max(resized_img.rows + y, 0), dst.rows);
  int u = (x >= 0) ? 0 : std::min(-x, resized_img.cols - 1);
  int v = (y >= 0) ? 0 : std::min(-y, resized_img.rows - 1);
  int px = std::max(x, 0);
  int py = std::max(y, 0);

  cv::Mat roi_dst = dst(cv::Rect(px, py, w, h));
  cv::Mat roi_resized = resized_img(cv::Rect(u, v, w, h));
  roi_resized.copyTo(roi_dst);
}

// 画像を画像に貼り付ける関数（サイズ指定を省略したバージョン）
void paste(cv::Mat dst, cv::Mat src, int x, int y) {
  paste(dst, src, x, y, src.rows, src.cols);
}

template <typename T>
bool ReplaceArea(
    const std::vector<T>& src_points, /* asset points */
    const std::vector<T>& dst_points, /* closed contours in image */
    const cv::Mat4b& asset, cv::Mat3b& replaced) {

  // scale asset to bb of detected points
  int dst_min_x = 1000000;
  int dst_min_y = 1000000;
  int dst_max_x = -1;
  int dst_max_y = -1;

  std::vector<cv::Point2f> f_src_points, f_dst_points;

  for (const auto& d : dst_points) {
    f_dst_points.push_back(cv::Point2f(d.x, d.y));

    dst_min_x = std::min(dst_min_x, int(d.x));
    dst_min_y = std::min(dst_min_y, int(d.y));
    dst_max_x = std::max(dst_max_x, int(d.x));
    dst_max_y = std::max(dst_max_y, int(d.y));

  }

  cv::Size dst_size(dst_max_x - dst_min_x, dst_max_y - dst_min_y);
  //cv::Mat4b tmp = asset;
  //cv::resize(asset, tmp, dst_size);
  cv::Mat4b scaled_asset = cv::Mat4b::zeros(replaced.size());
  paste(scaled_asset, asset, dst_min_x, dst_min_y, dst_max_x - dst_min_x,
        dst_max_y - dst_min_y);
  cv::imwrite("pasted.png", scaled_asset);

  float x_ratio = dst_size.width / static_cast<float>(asset.cols);
  float y_ratio = dst_size.height / static_cast<float>(asset.rows);
  for (const auto& s : src_points) {
    f_src_points.push_back(cv::Point2f(s.x * x_ratio + dst_min_x, s.y * y_ratio + dst_min_y));
  }

  auto tps = cv::createThinPlateSplineShapeTransformer();

  std::vector<cv::DMatch> matches;
  for (int i = 0; i < static_cast<int>(src_points.size()); i++) {
    matches.push_back(cv::DMatch(i, i, 0.0f));
  }
  //tps->estimateTransformation(f_src_points, f_dst_points, matches);
  tps->estimateTransformation(f_dst_points, f_src_points, matches);

  cv::Mat4b warped_asset = scaled_asset.clone();
  tps->warpImage(scaled_asset, warped_asset);

  cv::Mat1b warped_mask = cv::Mat1b::zeros(replaced.size());
  //warped_asset.forEach<cv::Vec4b>([&](cv::Vec4b& p, int* pos) -> void {
  //  if (p[3] == 0) {
  //    warped_mask.at<unsigned char>(pos[1], pos[0]) = 255;
  //  }
  //});
  std::vector<cv::Mat> planes;
  cv::split(warped_asset, planes);
  warped_mask = planes[3];

  cv::imwrite("warped_mask.png", warped_mask);

  cv::imwrite("warped.png", warped_asset);

  cv::Mat1b dst_mask = cv::Mat1b::zeros(replaced.size());

  std::vector<std::vector<cv::Point>> contours(1);
  contours[0] = dst_points;
  cv::drawContours(dst_mask, contours, 0, 255, -1);

  cv::imwrite("dst_mask.png", dst_mask);

  cv::Mat3b warped_asset_3b;
  cv::cvtColor(warped_asset, warped_asset_3b, cv::COLOR_BGRA2BGR);
  warped_asset_3b.copyTo(replaced, dst_mask);

  return true;
}

std::vector<std::string> Split(const std::string& input, char delimiter) {
  std::istringstream stream(input);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(stream, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}

}  // namespace

namespace aff {

struct Asset {
  std::string image_path;
  std::string points_path;
  std::string name;

  cv::Mat4b image;

  std::vector<int> dlib_indices;
  std::vector<cv::Point> points;
};

class AnimeFaceReplacerImpl {
 private:
  dlib::frontal_face_detector detector;
  dlib::shape_predictor pose_model;

  Asset mouse;

  bool LoadAssets(const Options& options);
  bool ReplaceLandmarks(const std::vector<cv::Point>& landmarks,
                        cv::Mat3b& replaced);

 public:
  bool Init(const Options& options);
  bool Replace(const cv::Mat3b& src, Output& output, const Options& options);
};

bool AnimeFaceReplacerImpl::LoadAssets(const Options& options) {
  // load mouse
  mouse.image_path = options.asset_dir + "/mouse.png";
  mouse.name = "mouse";
  mouse.image = cv::imread(mouse.image_path, cv::ImreadModes::IMREAD_UNCHANGED);
  if (mouse.image.empty()) {
    return false;
  }

  mouse.points.clear();
  mouse.dlib_indices.clear();

  mouse.points_path = options.asset_dir + "/mouse.txt";
  std::ifstream ifs(mouse.points_path);
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> splited = Split(line, ' ');
    if (splited.size() != 3) {
      printf("wrong format\n");
      return false;
    }

    mouse.dlib_indices.push_back(std::atoi(splited[0].c_str()));

    mouse.points.push_back(cv::Point(std::atoi(splited[1].c_str()),
                                     std::atoi(splited[2].c_str())));
  }

  return true;
}

bool AnimeFaceReplacerImpl::ReplaceLandmarks(
    const std::vector<cv::Point>& landmarks, cv::Mat3b& replaced) {
  std::vector<cv::Point> mouse_points;
  for (int i = 0; i < DLIB_MOUSE_OUTER_NUM; i++) {
    mouse_points.push_back(landmarks[DLIB_MOUSE_OUTER[i]]);
  }
  ReplaceArea(mouse.points, mouse_points, mouse.image, replaced);

  return true;
}

bool AnimeFaceReplacerImpl::Init(const Options& options) {
  detector = dlib::get_frontal_face_detector();
  try {
    dlib::deserialize(options.dlib_model_path) >> pose_model;
  } catch (dlib::serialization_error e) {
    return false;
  }

  return LoadAssets(options);
}

bool AnimeFaceReplacerImpl::Replace(const cv::Mat3b& src, Output& output,
                                    const Options& options) {
  cv::Mat3b tmp = src.clone();

  // Face and landmark detection
  DetectFace(detector, pose_model, tmp, output);
  output.vis_landmarks = src.clone();
  DrawDetectedFace(output.face_bb, output.landmarks, output.vis_landmarks);

  // Reduce color

  // Enhance contour

  // Uniform face color

  // Replace landmarks
  ReplaceLandmarks(output.landmarks, tmp);
  output.result = tmp;

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