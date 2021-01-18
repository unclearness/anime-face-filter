
#include <memory>
#include <vector>
#include "opencv2/core.hpp"

namespace aff {

static inline int DLIB_CHIN[17] = {0, 1,  2,  3,  4,  5,  6,  7, 8,
                                   9, 10, 11, 12, 13, 14, 15, 16};
static inline int DLIB_R_EYEBLOW[5] = {17, 18, 19, 20, 21};
static inline int DLIB_L_EYEBLOW[5] = {22, 23, 24, 25, 26};
static inline int DLIB_OUTER[27] = {0,  1,  2,  3,  4,  5,  6,  7,  8,
                                    9,  10, 11, 12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26};
static inline int DLIB_NOSE_UPPER[4] = {27, 28, 29, 30};
static inline int DLIB_NOSE_LOWER[5] = {31, 32, 33, 34, 35};
static inline int DLIB_R_EYE[6] = {36, 37, 38, 39, 40, 41};
static inline int DLIB_L_EYE[6] = {42, 43, 44, 45, 46, 47};
static inline int DLIB_MOUSE_OUTER_NUM = 12;
static inline int DLIB_MOUSE_OUTER[12] = {48, 49, 50, 51, 52, 53,
                                          54, 55, 56, 57, 58, 59};
static inline int DLIB_MOUSE_INNER[8] = {60, 61, 62, 63, 64, 65, 66, 67};

class DlibFaceDetectorImpl;
class DlibFaceDetector {
 private:
  std::unique_ptr<DlibFaceDetectorImpl> impl;

 public:
  bool Init(const std::string& dlib_model_path);
  bool Detect(cv::Mat3b& srcdst, cv::Rect& face_bb,
              std::vector<cv::Point>& landmarks);
  DlibFaceDetector();
  ~DlibFaceDetector();
};

}  // namespace aff