
#include <fstream>
#include <vector>

#include "aff/core.h"
#include "dlib.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/shape.hpp"

namespace aff {

struct Asset {
  std::string image_path;
  std::string points_path;
  std::string name;

  cv::Mat4b image;

  std::vector<int> dlib_indices;
  std::vector<cv::Point> points;
};
}  // namespace aff

namespace {

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

bool ReduceColor(cv::Mat3b& srcdst, unsigned char reduce_factor) {
  unsigned char reduce = 256 / reduce_factor;
  for (int j = 0; j < srcdst.rows; j++) {
    for (int i = 0; i < srcdst.cols; i++) {
      auto& p = srcdst.at<cv::Vec3b>(j, i);
      for (int k = 0; k < 3; k++) {
        p[k] = cv::saturate_cast<unsigned char>(p[k] / reduce *
                                                reduce);  // +num_color / 2);
      }
    }
  }

  return true;
}

// Enhance contour
bool EnhanceContour() { return true; }

// Uniform face color
bool UniformFaceColor() { return true; }

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
  // cv::Mat4b tmp = asset;
  // cv::resize(asset, tmp, dst_size);
  cv::Mat4b scaled_asset = cv::Mat4b::zeros(replaced.size());
  paste(scaled_asset, asset, dst_min_x, dst_min_y, dst_max_x - dst_min_x,
        dst_max_y - dst_min_y);
  cv::imwrite("pasted.png", scaled_asset);

  float x_ratio = dst_size.width / static_cast<float>(asset.cols);
  float y_ratio = dst_size.height / static_cast<float>(asset.rows);
  for (const auto& s : src_points) {
    f_src_points.push_back(
        cv::Point2f(s.x * x_ratio + dst_min_x, s.y * y_ratio + dst_min_y));
  }

  auto tps = cv::createThinPlateSplineShapeTransformer();

  std::vector<cv::DMatch> matches;
  for (int i = 0; i < static_cast<int>(src_points.size()); i++) {
    matches.push_back(cv::DMatch(i, i, 0.0f));
  }
  // tps->estimateTransformation(f_src_points, f_dst_points, matches);
  tps->estimateTransformation(f_dst_points, f_src_points, matches);

  cv::Mat4b warped_asset = scaled_asset.clone();
  tps->warpImage(scaled_asset, warped_asset);

  cv::Mat1b warped_mask = cv::Mat1b::zeros(replaced.size());
  // warped_asset.forEach<cv::Vec4b>([&](cv::Vec4b& p, int* pos) -> void {
  //  if (p[3] == 0) {
  //    warped_mask.at<unsigned char>(pos[1], pos[0]) = 255;
  //  }
  //});

  std::vector<cv::Mat> planes;
  cv::split(warped_asset, planes);
  warped_mask = (planes[3] == 255);

  cv::imwrite("warped_mask.png", warped_mask);

  cv::imwrite("warped.png", warped_asset);

  cv::Mat1b dst_mask = cv::Mat1b::zeros(replaced.size());

  {
    std::vector<std::vector<cv::Point>> contours(1);
    contours[0] = dst_points;
    cv::drawContours(dst_mask, contours, 0, 255, -1);
  }
  cv::imwrite("dst_mask.png", dst_mask);

  cv::Mat3b warped_asset_3b;

  cv::cvtColor(warped_asset, warped_asset_3b, cv::COLOR_BGRA2BGR);

  cv::Mat1b final_mask = dst_mask & warped_mask;

  cv::imwrite("final_mask.png", final_mask);

  // warped_asset_3b.copyTo(replaced, final_mask);
  cv::Moments mu = cv::moments(final_mask, true);
  cv::Point object_p(mu.m10 / mu.m00, mu.m01 / mu.m00);
  // printf("%d, %d\n", object_p.x, object_p.y);
  cv::seamlessClone(warped_asset_3b, replaced.clone(), final_mask, object_p,
                    replaced, cv::NORMAL_CLONE);

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

bool DumpAsset(const std::string& debug_dir, const aff::Asset& asset) {
  cv::Mat4b debug_image = asset.image.clone();

  for (auto i = 0; i < asset.dlib_indices.size(); i++) {
    const auto& dlib_index = asset.dlib_indices[i];
    const auto& point = asset.points[i];
    cv::circle(debug_image, point, 3, cv::Scalar(0, 0, 255, 255), -1);
    cv::putText(debug_image, std::to_string(dlib_index), point,
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0, 255), 2);
  }

  cv::imwrite(debug_dir + asset.name + ".png", debug_image);

  return true;
}

}  // namespace

namespace aff {

class AnimeFaceReplacerImpl {
 private:
  DlibFaceDetector dlib_face_detector;

  Asset mouse;

  Options options_;

  bool LoadAssets(const Options& options);
  bool DumpAssets(const std::string& debug_dir);
  bool ReplaceLandmarks(const std::vector<cv::Point>& landmarks,
                        cv::Mat3b& replaced);

 public:
  bool Init(const Options& options);
  bool Replace(const cv::Mat3b& src, Output& output, const Options& options);
};

bool AnimeFaceReplacerImpl::DumpAssets(const std::string& debug_dir) {
  DumpAsset(debug_dir, mouse);

  return true;
}

bool AnimeFaceReplacerImpl::LoadAssets(const Options& options) {
  options_ = options;

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
  DumpAssets(options_.debug_dir);

  std::vector<cv::Point> mouse_points;
  for (int i = 0; i < DLIB_MOUSE_OUTER_NUM; i++) {
    mouse_points.push_back(landmarks[DLIB_MOUSE_OUTER[i]]);
  }
  ReplaceArea(mouse.points, mouse_points, mouse.image, replaced);

  return true;
}

bool AnimeFaceReplacerImpl::Init(const Options& options) {
  options_ = options;

  if (!dlib_face_detector.Init(options.dlib_model_path)) {
    return false;
  }

  return LoadAssets(options);
}

bool AnimeFaceReplacerImpl::Replace(const cv::Mat3b& src, Output& output,
                                    const Options& options) {
  cv::Mat3b tmp = src.clone();

  // Face and landmark detection
  dlib_face_detector.Detect(tmp, output.face_bb, output.landmarks);

  output.vis_landmarks = src.clone();
  DrawDetectedFace(output.face_bb, output.landmarks, output.vis_landmarks);

  // Reduce color
  // ReduceColor(tmp, 8);

  // Enhance contour

  // Uniform face color

  // Replace landmarks
  ReplaceLandmarks(output.landmarks, tmp);

  // Blur
  // cv::GaussianBlur(tmp, tmp, cv::Size(9, 9), 1.0);

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