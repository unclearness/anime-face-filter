/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "aff/core.h"
#include "aff/timer.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

//#define USE_VIDEO

int main() {
#ifdef USE_VIDEO
  cv::VideoCapture cap("../data/test/trump.mp4");
#else
  cv::VideoCapture cap(0);
#endif

  if (!cap.isOpened()) {
    return -1;
  }

  aff::Timer timer;
  aff::Output output;
  aff::Options options;
  aff::AnimeFaceReplacer replacer;
  timer.Start();
  replacer.Init(options);
  timer.End();
  std::cout << "Init: " << timer.elapsed_msec() << "ms" << std::endl;
  int w = 240;

  cv::Mat frame, latest_result;
  latest_result = cv::Mat3b::zeros(240, 240);

  while (cap.read(frame)) {
    cv::imshow("src", frame);

    auto org_size = frame.size();
    int h = static_cast<int>(w * static_cast<float>(frame.rows) /
                             static_cast<float>(frame.cols));

    cv::resize(frame, frame, cv::Size(w, h), 0, 0,
               cv::InterpolationFlags::INTER_NEAREST);

    timer.Start();
    bool ret = replacer.Replace(frame, output, options);
    timer.End();
    std::cout << "Replace: " << timer.elapsed_msec() << "ms" << std::endl;

    if (ret) {
      cv::resize(output.result, latest_result, org_size);
    }

    cv::imshow("anime face", latest_result);

    const int key = cv::waitKey(1);
    if (key == 'q') {
      break;
    } else if (key == 's') {
      cv::imwrite("src.png", frame);
      cv::imwrite("anime-face.png", latest_result);
    }
  }
  cv::destroyAllWindows();
  return 0;
}