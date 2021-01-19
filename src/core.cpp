
#include <fstream>
#include <numeric>
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

std::vector<cv::Point> ExpandFacePart(
    const std::vector<cv::Point>& landmarks,
    cv::Vec2f xvec = cv::Vec2f(1.0f, 0.0f), float xratio = 1.3f,
    cv::Vec2f yvec = cv::Vec2f(0.0f, 1.0f), float yratio = 1.3f, int min_x = -1,
    int max_x = std::numeric_limits<int>::max(), int min_y = -1,
    int max_y = std::numeric_limits<int>::max()) {
  // Get centroid
  auto centroid =
      std::accumulate(landmarks.begin(), landmarks.end(), cv::Point(0, 0)) /
      static_cast<int>(landmarks.size());
  auto centroid_v = cv::Vec2f(centroid.x, centroid.y);

  // Expand distance from centroid
  std::vector<cv::Point> expanded_landmarks;
  std::transform(landmarks.begin(), landmarks.end(),
                 std::back_inserter(expanded_landmarks),
                 [&](const auto& org_p) {
                   auto direc_p = org_p - centroid;
                   if (direc_p.x == 0 && direc_p.y == 0) {
                        return org_p;
                   }
                   auto direc = cv::Vec2f(direc_p.x, direc_p.y);
                   auto dist = std::max(cv::norm(direc),1.0);
                   auto normed_direc = cv::normalize(direc);
                   auto x_len = xvec.dot(normed_direc) * dist * xratio;
                   auto y_len = yvec.dot(normed_direc) * dist * yratio;
                   cv::Point p = centroid_v + xvec * x_len + yvec * y_len;
                   p.x = std::clamp(p.x, min_x, max_x);
                   p.y = std::clamp(p.y, min_y, max_y);
                   return p;
                 });
  return expanded_landmarks;
}

std::vector<cv::Point> GetFacePartLandmarks(
    const std::vector<cv::Point>& all_landmarks, int part_num,
    int* part_indices) {
  std::vector<cv::Point> part_landmarks;
  for (int i = 0; i < part_num; i++) {
    part_landmarks.push_back(all_landmarks[part_indices[i]]);
  }
  return part_landmarks;
}

bool ReplaceFacePartLandmarks(std::vector<cv::Point>& all_landmarks,
                              int part_num, int* part_indices,
                              const std::vector<cv::Point>& part_landmarks) {
  for (int i = 0; i < part_num; i++) {
    all_landmarks[part_indices[i]] = part_landmarks[i];
  }
  return true;
}

std::vector<cv::Point> GetMouseLandmarks(
    const std::vector<cv::Point>& all_landmarks) {
  return GetFacePartLandmarks(all_landmarks, aff::DLIB_MOUSE_OUTER_NUM,
                              aff::DLIB_MOUSE_OUTER);
}

std::vector<cv::Point> GetREyeLandmarks(
    const std::vector<cv::Point>& all_landmarks) {
  return GetFacePartLandmarks(all_landmarks, aff::DLIB_EYE_NUM,
                              aff::DLIB_R_EYE);
}

bool ExpandMouse(std::vector<cv::Point>& landmarks) {
  auto expanded = ExpandFacePart(GetMouseLandmarks(landmarks));
  ReplaceFacePartLandmarks(landmarks, aff::DLIB_MOUSE_OUTER_NUM,
                           aff::DLIB_MOUSE_OUTER, expanded);
  return true;
}

bool ExpandREye(std::vector<cv::Point>& landmarks) {
  cv::Point2f xdiff = landmarks[36] - landmarks[39];
  cv::Vec2f xvec = cv::normalize(cv::Vec2f(xdiff.x, xdiff.y));

  cv::Vec2f yvec(-xvec[1], xvec[0]);

  auto expanded =
      ExpandFacePart(GetREyeLandmarks(landmarks), xvec, 1.3f, yvec, 3.0f);
  ReplaceFacePartLandmarks(landmarks, aff::DLIB_EYE_NUM, aff::DLIB_R_EYE,
                           expanded);

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

#if 1
  warped_asset_3b.copyTo(replaced, final_mask);
#else
  cv::Moments mu = cv::moments(final_mask, true);
  cv::Point object_p(mu.m10 / mu.m00, mu.m01 / mu.m00);
  printf("%d, %d\n", object_p.x, object_p.y);
  cv::seamlessClone(warped_asset_3b, replaced.clone(), final_mask, object_p,
                    replaced, cv::NORMAL_CLONE);
#endif

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
  std::string mouse_basename = "mouse";
  Asset r_eye_;
  std::string reye_basename = "r_eye";

  Options options_;

  bool LoadAsset(Asset& asset, const std::string& basename,
                 const Options& options);
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
  DumpAsset(debug_dir, r_eye_);
  return true;
}

bool AnimeFaceReplacerImpl::LoadAsset(Asset& asset, const std::string& basename,
                                      const Options& options) {
  asset.image_path = options.asset_dir + "/" + basename + ".png";
  asset.name = basename;
  asset.image = cv::imread(asset.image_path, cv::ImreadModes::IMREAD_UNCHANGED);
  if (asset.image.empty()) {
    return false;
  }

  asset.points.clear();
  asset.dlib_indices.clear();

  asset.points_path = options.asset_dir + "/" + basename + ".txt";
  std::ifstream ifs(asset.points_path);
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> splited = Split(line, ' ');
    if (splited.size() != 3) {
      printf("wrong format\n");
      return false;
    }

    asset.dlib_indices.push_back(std::atoi(splited[0].c_str()));

    asset.points.push_back(cv::Point(std::atoi(splited[1].c_str()),
                                     std::atoi(splited[2].c_str())));
  }

  return true;
}

bool AnimeFaceReplacerImpl::LoadAssets(const Options& options) {
  options_ = options;

  LoadAsset(mouse, mouse_basename, options);
  LoadAsset(r_eye_, reye_basename, options);

  return true;
}

bool AnimeFaceReplacerImpl::ReplaceLandmarks(
    const std::vector<cv::Point>& landmarks, cv::Mat3b& replaced) {
  DumpAssets(options_.debug_dir);

  ReplaceArea(mouse.points, GetMouseLandmarks(landmarks), mouse.image,
              replaced);
  ReplaceArea(r_eye_.points, GetREyeLandmarks(landmarks), r_eye_.image,
              replaced);

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
  cv::imwrite("detected_org.png", output.vis_landmarks);

  // Reduce color
  ReduceColor(tmp, 8);

  // Enhance contour

  // Uniform face color

  // Expand face parts
  ExpandMouse(output.landmarks);
  ExpandREye(output.landmarks);
  output.vis_landmarks = src.clone();
  DrawDetectedFace(output.face_bb, output.landmarks, output.vis_landmarks);

  // Replace landmarks
  ReplaceLandmarks(output.landmarks, tmp);

  // Blur
  // cv::GaussianBlur(tmp, tmp, cv::Size(9, 9), 3.0);

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