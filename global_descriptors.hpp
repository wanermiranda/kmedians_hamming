#ifndef _IMG_GD_H_
#define _IMG_GD_H_
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "bag_of_words.hpp"
#include <exception>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>

#include <unistd.h>


	void extractlist_SIFT_1internal(std::string dataset_file, std::string output_dir);

	void extractlist_ORB_1internal(std::string dataset_file, std::string output_dir, size_t features_size);

	void extractlist_HOG_1internal(std::string dataset_file, std::string output_dir);

	void extractlist_LATCH_1internal(std::string dataset_file, std::string output_dir);



#endif
