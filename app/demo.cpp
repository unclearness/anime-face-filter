#include "aff/core.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

int main() {
  std::string lena_path = "../data/test/trump.jpg";

  cv::Mat3b src = cv::imread(lena_path, cv::ImreadModes::IMREAD_COLOR);
  aff::Output output;
  aff::Options options;

  aff::AnimeFaceReplacer replacer;
  replacer.Init(options);
  replacer.Replace(src, output, options);

  cv::imwrite("detected.png", output.vis_landmarks);
  cv::imwrite("result.png", output.result);

  return 0;
}