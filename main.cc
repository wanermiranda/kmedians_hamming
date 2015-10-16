/*
 * main.cpp
 *
 *  Created on: Jul 22, 2015
 *      Author: waner
 */
#include <stdio.h>
#include <sys/stat.h>
#include <string>
#include "bag_of_words.hpp"
#include "global_descriptors.hpp"

using namespace std;
int main(int argv, char **args) {
	string path;
	string output;
	string task;
	string desc;
	size_t keypoints_size;

	if (argv == 1) {
		path = "/home/waner/git-new/kmedians_hamming/bigvoc.txt";
		desc = "hog";
		task = "dictionary";
//		task = "features";
		output = "dictionary_hog.yml";
		keypoints_size = 300;
		output = "./hog/";
	} else {
		path = args[1];
		output = args[2];
		task = args[3];
		desc = args[4];
		keypoints_size = atoi(args[5]);
	}

	printf("Reading dataset %s \n", path.c_str());

	if (task == "dictionary") {
		if (desc == "sift")
			buildDictionary_SIFT(path, output);
		else if (desc == "orb")
			buildDictionary_ORB(path, output);
		else if (desc == "hog")
			buildDictionary_ORB(path, output);
		else
			buildDictionary_LATCH2(path, output);
	}
	else {
		mkdir(output.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (desc == "sift")
			extractlist_SIFT_1internal(path, output);
		else if (desc == "orb")
			extractlist_ORB_1internal(path, output, keypoints_size);
		else if (desc == "hog")
			extractlist_HOG_1internal(path, output);
		else if (desc == "latch")
			extractlist_LATCH_1internal(path, output);
	}
	return 0;
}

