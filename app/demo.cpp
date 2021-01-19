/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "aff/core.h"
#include "aff/timer.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

int main() {
  std::string lena_path = "../data/test/lena.jpg";
  aff::Timer timer;
  cv::Mat3b src = cv::imread(lena_path, cv::ImreadModes::IMREAD_COLOR);

  int w = 240;
  int h = static_cast<int>(w * static_cast<float>(src.rows) /
                           static_cast<float>(src.cols));
  cv::resize(src, src, cv::Size(w, h));
  aff::Output output;
  aff::Options options;
  options.debug_dir = "./";

  aff::AnimeFaceReplacer replacer;
  timer.Start();
  replacer.Init(options);
  timer.End();
  std::cout << "Init: " << timer.elapsed_msec() << "ms" << std::endl;
  timer.Start();
  replacer.Replace(src, output, options);
  timer.End();
  std::cout << "Replace: " << timer.elapsed_msec() << "ms" << std::endl;

  if (!options.debug_dir.empty()) {
    cv::imwrite(options.debug_dir + "detected.png", output.vis_landmarks);
    cv::imwrite(options.debug_dir + "result.png", output.result);
  }
  return 0;
}