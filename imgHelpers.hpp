#ifndef _IMG_HELPERS_H_
#define _IMG_HELPERS_H_
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <vector>

namespace ih {
	 const char* const MNT_SDCARD_DICTIONARY_ORB_YML = "dictionary_orb.yml";
	 const char* const MNT_SDCARD_DICTIONARY_SIFT_YML = "dictionary_sift.yml";
	 const char* const MNT_SDCARD_DICTIONARY_HOG_YML = "dictionary_hog.yml";
	 const char* const MNT_SDCARD_DICTIONARY_LATCH_YML = "dictionary_latch.yml";

	 const int FAST_TRESHOLD = 40;
	 const float PI_F = 3.14159265358979f;
	 const int DICTIONARY_SIZE_ORB = 1024;
	 const int MINIMUM_KEYPOINTS = 64;
	 const int MAXIMUM_KEYPOINTS = 384;
	 const int DICTIONARY_SIZE_SIFT = 256;
	 const int DICTIONARY_SIZE_LATCH = 256;
	 const int DICTIONARY_SIZE_HOG = 256;


	 const float DENSE_IFS       = 15.0f; // Initial feature scale
	 const int DENSE_FSL       = 1;     // Feature scale levels
	 const float DENSE_FSM       = 0.1f;  // The level parameters are multiplied by fsm
	 const int DENSE_XY_STEP   = 6;     // The dense sampling samples at every "xy-step" pixel in the image
	 const int DENSE_IIB       = 0;     // Initial image bound
	 const bool DENSE_V_XY_SWS  = true;  // Vary "xy-step" with scale
	 const bool DENSE_V_IMG_BWS = false; // Vary image bound with scale"
}


cv::Mat loadScaledImageSqrd(const char *path, int flags,
		float target_width = 300.0);

cv::Mat loadScaledImage(const char *path, int flags, float maxSize = 500.0);

cv::Mat loadImage(const char *path, int flags);

int getMinorDimension(cv::Mat img_original);

#endif
