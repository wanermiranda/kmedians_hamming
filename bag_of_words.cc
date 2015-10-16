#include "bag_of_words.hpp"
#include "precomp.hpp"
#include "KmediansBinary.h"
using namespace cv;


void buildDictionary_ORB(string dataset_file, string dictionary_out) {
	DenseFeatureDetector detector(ih::DENSE_IFS, ih::DENSE_FSL, ih::DENSE_FSM, ih::DENSE_XY_STEP, ih::DENSE_IIB, ih::DENSE_V_XY_SWS, ih::DENSE_V_IMG_BWS);
	OrbDescriptorExtractor extractor;
	Mat allDescriptors(0, 0, CV_32F);

	try {
		std::ifstream ifs(dataset_file.c_str());

		std::string path;

		while (std::getline(ifs, path)) {
			Mat descriptors;
			vector<KeyPoint> keypoints;

			Mat img = imread(path.c_str()); //loadScaledImage(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			printf("Image: %s \n", path.c_str());

			detector.detect(img, keypoints);
			printf("Keypoints collected %zu\n", keypoints.size());
//			KeyPointsFilter::runByImageBorder(keypoints,img.size(),getMinorDimension(img)*.45);
			printf("Keypoints filtered %zu\n", keypoints.size());
			extractor.compute(img, keypoints, descriptors);

			allDescriptors.push_back(descriptors);

			// release section
			img.release();
			descriptors.release();
		}

		if (allDescriptors.type() != CV_32F) {
			allDescriptors.convertTo(allDescriptors, CV_32F);
		}

		printf("Creating BoVW");
		TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
		int retries = 1;
		int flags = KMEANS_RANDOM_CENTERS;

//		BOWKMeansTrainer bowTrainer(ih::DICTIONARY_SIZE_ORB, tc, retries, flags);
		//convert featuresUnclustered to type CV_32F
		Mat featuresUnclusteredF(allDescriptors.rows, allDescriptors.cols,
		CV_32F);
		allDescriptors.convertTo(featuresUnclusteredF, CV_32F);
		//cluster the feature vectors
		Mat labels;
		Mat dictionary;

//        KmediansBinary(FEATURES
//		Mat dictionary = bowTrainer.cluster(featuresUnclusteredF);

		FileStorage fs(dictionary_out, FileStorage::WRITE);
		fs << "dictionary" << dictionary;

		fs.release();

	} catch (const std::exception & e) {
		printf("Exception %s", e.what());
	}

	// release section
	allDescriptors.release();
}

void buildDictionary_SIFT(string dataset_file, string dictionary_out) {
	FastFeatureDetector detector;
	SiftDescriptorExtractor descriptor;
	Mat allDescriptors(0, 0, CV_32F);

	try {
		std::ifstream ifs(dataset_file.c_str());

		std::string path;

		while (std::getline(ifs, path)) {
			Mat descriptors;
			vector<KeyPoint> keypoints;

			Mat img = loadScaledImage(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			printf("Image: %s \n", path.c_str());

			detector.detect(img, keypoints);

			KeyPointsFilter::retainBest(keypoints, ih::MAXIMUM_KEYPOINTS);

			descriptor.compute(img, keypoints, descriptors);

			allDescriptors.push_back(descriptors);
			std::cout << "Keypoints: " << keypoints.size() << " Total: " << allDescriptors.size() << std::endl;


			// release section
			img.release();
			descriptors.release();
		}

		if (allDescriptors.type() != CV_32F) {
			allDescriptors.convertTo(allDescriptors, CV_32F);
		}

		printf("Creating BoVW");
		TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
		int retries = 1;
		int flags = KMEANS_RANDOM_CENTERS;
		BOWKMeansTrainer bowTrainer(ih::DICTIONARY_SIZE_SIFT, tc, retries, flags);
		//convert featuresUnclustered to type CV_32F
		Mat featuresUnclusteredF(allDescriptors.rows, allDescriptors.cols,
		CV_32F);
		allDescriptors.convertTo(featuresUnclusteredF, CV_32F);
		//cluster the feature vectors
		Mat dictionary = bowTrainer.cluster(featuresUnclusteredF);

		FileStorage fs(dictionary_out, FileStorage::WRITE);
		fs << "dictionary" << dictionary;

		fs.release();

	} catch (const std::exception & e) {
		printf("Exception %s", e.what());
	}

	// release section
	allDescriptors.release();

}


void buildDictionary_LATCH2(string dataset_file, string dictionary_out)
{
	int bytes = 32; bool rotationInvariance = true; int half_ssd_size = 3;
	FastFeatureDetector detector;
	features2d::LATCHDescriptorExtractorImpl descriptor(bytes, rotationInvariance, half_ssd_size);;
	vector<KeyPoint> keypoints;
	Mat allDescriptors;

	try {
			std::ifstream ifs(dataset_file.c_str());

			std::string path;
			int count = 0;
			while (std::getline(ifs, path)) {
//				if (++count > 10) break;
				Mat descriptors;
				vector<KeyPoint> keypoints;

				Mat img = loadScaledImage(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
				printf("Image: %s \n", path.c_str());

				detector.detect(img, keypoints);
				KeyPointsFilter::removeDuplicated(keypoints);
				KeyPointsFilter::retainBest(keypoints, ih::MAXIMUM_KEYPOINTS);

				descriptor.compute(img, keypoints, descriptors);

				allDescriptors.push_back(descriptors);
				std::cout << "Keypoints: " << keypoints.size() << " Total: " << allDescriptors.size() << std::endl;


				// release section
				img.release();
				descriptors.release();
			}


			printf("Creating BoVW");
			TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
			int retries = 3;
			int flags = KMEANS_RANDOM_CENTERS;

			cv::Mat uDictionary;
			Mat labels;

			KmediansBinary cluster(allDescriptors,ih::DICTIONARY_SIZE_LATCH, retries, uDictionary);


			FileStorage fs(dictionary_out, FileStorage::WRITE);
			fs << "dictionary" << uDictionary;

			fs.release();

		} catch (const std::exception & e) {
			printf("Exception %s", e.what());
		}

		// release section
		allDescriptors.release();
}

void buildDictionary_LATCH(string dataset_file, string dictionary_out)
{
	int bytes = 32; bool rotationInvariance = true; int half_ssd_size = 3;
	FastFeatureDetector detector;
	features2d::LATCHDescriptorExtractorImpl descriptor(bytes, rotationInvariance, half_ssd_size);;
	vector<KeyPoint> keypoints;
	Mat allDescriptors;

	try {
			std::ifstream ifs(dataset_file.c_str());

			std::string path;

			while (std::getline(ifs, path)) {
				Mat descriptors;
				vector<KeyPoint> keypoints;

				Mat img = loadScaledImage(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
				printf("Image: %s \n", path.c_str());

				detector.detect(img, keypoints);
				KeyPointsFilter::removeDuplicated(keypoints);
				KeyPointsFilter::retainBest(keypoints, ih::MAXIMUM_KEYPOINTS);

				descriptor.compute(img, keypoints, descriptors);

				allDescriptors.push_back(descriptors);
				std::cout << "Keypoints: " << keypoints.size() << " Total: " << allDescriptors.size() << std::endl;


				// release section
				img.release();
				descriptors.release();
			}

//			if (allDescriptors.type() != CV_32F) {
//				allDescriptors.convertTo(allDescriptors, CV_32F);
//			}

			printf("Creating BoVW");
			TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
			int retries = 1;
			int flags = KMEANS_RANDOM_CENTERS;
			BOWKMeansTrainer bowTrainer(ih::DICTIONARY_SIZE_LATCH, tc, retries, flags);
			//convert featuresUnclustered to type CV_32F
			Mat featuresUnclusteredF(allDescriptors.rows, allDescriptors.cols,
					CV_32F);
			allDescriptors.convertTo(featuresUnclusteredF, CV_32F);
			//cluster the feature vectors
			Mat dictionary = bowTrainer.cluster(featuresUnclusteredF);
			cv::Mat uDictionary;

			dictionary.convertTo(uDictionary, CV_32F);

			FileStorage fs(dictionary_out, FileStorage::WRITE);
			fs << "dictionary" << uDictionary;

			fs.release();

		} catch (const std::exception & e) {
			printf("Exception %s", e.what());
		}

		// release section
		allDescriptors.release();
}

void buildDictionary_HOG(string dataset_file, string dictionary_out) {
	HOGDescriptor descriptor( Size(32,32), Size(16,16), Size(16,16), Size(8,8), 9);
//	HOGDescriptor descriptor;
	FastFeatureDetector detector;
	vector<KeyPoint> keypoints;
	Mat allDescriptors(0, 0, CV_32F);

	try {
		std::ifstream ifs(dataset_file.c_str());

		std::string path;

		while (std::getline(ifs, path)) {

			Mat img = loadScaledImage(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE,
					300);
			printf("Image: %s \n", path.c_str());
			detector.detect(img, keypoints);
			KeyPointsFilter::removeDuplicated(keypoints);
			KeyPointsFilter::runByImageBorder(keypoints, img.size(), 16);
			KeyPointsFilter::retainBest(keypoints, ih::MAXIMUM_KEYPOINTS);
			for (KeyPoint kp : keypoints) {
				vector<float> descriptors;
				vector<Point> locations;
				Mat imgCut(32, 32, CV_8U);
				int pad = 32 /2;
				img(Rect(kp.pt.x-pad, kp.pt.y-pad, kp.size, kp.size)).copySize(imgCut);
//				descriptor.compute(imgCut, descriptors, Size(0, 0), Size(0, 0), locations);
				descriptor.compute(imgCut, descriptors);
//				std::cout << "Descriptors size: " << descriptors.size() << std::endl;

				Mat dctmat = Mat(descriptors).t();


				allDescriptors.push_back(dctmat);
			}
			std::cout << "Keypoints: " << keypoints.size() << " Total: " << allDescriptors.size() << std::endl;

			// release section
			img.release();
		}

		printf("Converting vectors");
		if (allDescriptors.type() != CV_32F) {
			allDescriptors.convertTo(allDescriptors, CV_32F);
		}

		printf("Creating BoVW");
		TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
		int retries = 1;
		int flags = KMEANS_RANDOM_CENTERS;
		BOWKMeansTrainer bowTrainer(ih::DICTIONARY_SIZE_SIFT, tc, retries, flags);
		//convert featuresUnclustered to type CV_32F
		Mat featuresUnclusteredF(allDescriptors.rows, allDescriptors.cols,
		CV_32F);
		allDescriptors.convertTo(featuresUnclusteredF, CV_32F);
		//cluster the feature vectors
		Mat dictionary = bowTrainer.cluster(featuresUnclusteredF);

		FileStorage fs(dictionary_out, FileStorage::WRITE);
		fs << "dictionary" << dictionary;

		fs.release();

	} catch (const std::exception & e) {
		printf("Exception %s", e.what());
	}

	// release section
	allDescriptors.release();

}

