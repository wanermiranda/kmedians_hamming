#ifndef _BOW_H_
#define _BOW_H_
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include "global_descriptors.hpp"
#include "imgHelpers.hpp"



void buildDictionary_ORB(std::string dataset_file, std::string dictionary_out);
void buildDictionary_SIFT(std::string dataset_file, std::string dictionary_out);
void buildDictionary_HOG(std::string dataset_file, std::string dictionary_out);
void buildDictionary_LATCH(std::string dataset_file, std::string dictionary_out);

void buildDictionary_LATCH2(std::string dataset_file, std::string dictionary_out);

#endif
