#include "aff/core.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

int main() {
  std::string lena_path = "../data/test/lena.jpg";

  cv::Mat3b src = cv::imread(lena_path, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat3b dst;
  aff::Options options;

  aff::AnimeFaceReplacer replacer;
  replacer.Init(options);
  replacer.Replace(src, dst, options);

  return 0;
}