#include "global_descriptors.hpp"
#include "precomp.hpp"

using namespace std;
using namespace cv;

string SplitFilename(const std::string& str) {
	unsigned found = str.find_last_of("/\\");
	return str.substr(found + 1);
}
string printFloatVar(float* floatVar, int size) {
	std::string str_vector = "";

	for (int idx = 0; idx < size; idx++) {
		char value[50];
		sprintf(value, "%f", floatVar[idx]);
		str_vector += value;
		if (idx != (size - 1))
			str_vector += " ";
	}

	str_vector += "";

	return str_vector;

}

Mat loadScaledImage_desc(const char *path, int flags) {
	Mat img_original = imread(path, flags);

	Mat img;

	int originalSize =
			(img_original.size().height > img_original.size().width) ?
					img_original.size().height : img_original.size().width;

	double ratio = (originalSize > 500.0) ? (500.0 / originalSize) : 1.0;

	resize(img_original, img, Size(), ratio, ratio, INTER_LINEAR);
	img_original.release();

	return img;
}

Mat getOrbFeatures(const char* TAG, const DenseFeatureDetector& detector,
		FeatureDetector& featDetector, BOWImgDescriptorExtractor bowDE,
		const char* path, size_t features_size) {
	vector<KeyPoint> keypoints;
	Mat img = loadScaledImage_desc(path, CV_LOAD_IMAGE_GRAYSCALE);
	cout << "Running dense detection on " << path << endl;
	cout << "Image properties" << img.cols << "x" << img.rows << endl;
	featDetector.detect(img, keypoints);

	cout << "Keypoints before: " << keypoints.size() << endl;
	KeyPointsFilter::removeDuplicated(keypoints);
	KeyPointsFilter::retainBest(keypoints, features_size);
	cout << "Keypoints after filters: " << keypoints.size() << endl;

	if (keypoints.size() < features_size) {
		int missing = features_size - keypoints.size();
		vector<KeyPoint> tempKeypoints;
		detector.detect(img, tempKeypoints);
		if (missing > tempKeypoints.size())
			missing = tempKeypoints.size();
		for (int i = 0; i < missing; i++)
			std::swap(tempKeypoints[i],
					tempKeypoints[i + (std::rand() % (missing - i))]);

		for (int i = 0; i < missing; i++)
			keypoints.push_back(tempKeypoints[i]);
	} else if (keypoints.size() > features_size)
		keypoints.resize(features_size);

	cout << "Keypoints after filling: " << keypoints.size() << endl;

	cout << "Running ORB on " << path << endl;
	Mat matDescriptor = Mat(0, ih::DICTIONARY_SIZE_ORB, CV_32F);
	//extract BoW (or BoF) descriptor from given image
	bowDE.compute(img, keypoints, matDescriptor);
	img.release();
	return matDescriptor;
}

void extractlist_ORB_1internal(string dataset_file, string output_dir,
		size_t features_size) {
	const char* TAG = "jni-goldenretrieval";

	DenseFeatureDetector detector(ih::DENSE_IFS, ih::DENSE_FSL, ih::DENSE_FSM,
			ih::DENSE_XY_STEP, ih::DENSE_IIB, ih::DENSE_V_XY_SWS,
			ih::DENSE_V_IMG_BWS);
	FastFeatureDetector fastDetector(ih::FAST_TRESHOLD);
	Ptr<DescriptorExtractor> extractor(new OrbDescriptorExtractor());
	Ptr<DescriptorMatcher> matcher(new BFMatcher(NORM_HAMMING));

	cout << "Loading dictionary" << endl;

	// Loading bag of words from file
	Mat dictionaryF;
	FileStorage fs(ih::MNT_SDCARD_DICTIONARY_ORB_YML, FileStorage::READ);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	fs["dictionary"] >> dictionaryF;
	fs.release();

	Mat dictionary;
	cout << "Dictionary sizes " << dictionaryF.cols << dictionaryF.rows << endl;
	dictionaryF.convertTo(dictionary, CV_8U);
	bowDE.setVocabulary(dictionary);

	cout << "Iterating over images" << endl;

	try {
		std::ifstream ifs(dataset_file.c_str());

		std::string path;

		while (std::getline(ifs, path)) {
			Mat matFeature = getOrbFeatures(TAG, detector, fastDetector, bowDE,
					path.c_str(), features_size);
			std::ofstream feature_file;
			string output_file = output_dir + SplitFilename(path);
			output_file = output_file.replace(output_file.find("jpg"), 3,
					"bin");
			feature_file.open(output_file.c_str());
			float * floatFeature = (float *) matFeature.data;
			feature_file
					<< printFloatVar(floatFeature, ih::DICTIONARY_SIZE_SIFT);

			matFeature.release();
			feature_file.close();

		}
	} catch (const std::exception & e) {
		cout << "Exception " << e.what() << endl;
	}

}

Mat getSIFTFeatures(const char* TAG, FastFeatureDetector& fastDetector,
		BOWImgDescriptorExtractor bowDE, const char* path) {
	vector<KeyPoint> keypoints;
	Mat img = loadScaledImage_desc(path, CV_LOAD_IMAGE_GRAYSCALE);
	printf("\nRunning detection on %s", path);
	printf("\nImage properties %d x %d", img.cols, img.rows);
	fastDetector.detect(img, keypoints);
	printf("\nKeypoints before %zu", keypoints.size());
	KeyPointsFilter::retainBest(keypoints, ih::MAXIMUM_KEYPOINTS);
	printf("\nKeypoints after %zu", keypoints.size());

	printf("\nRunning SIFT on %s", path);
	Mat matDescriptor = Mat(0, ih::DICTIONARY_SIZE_SIFT, CV_32F);
	//extract BoW (or BoF) descriptor from given image
	bowDE.compute(img, keypoints, matDescriptor);
	float* floatDescriptor = (float*) (matDescriptor.data);

	img.release();

	return matDescriptor;
}

void extractlist_SIFT_1internal(string dataset_file, string output_dir) {
	const char* TAG = "jni-goldenretrieval";

	FastFeatureDetector fastDetector;
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor());
	Ptr<DescriptorMatcher> matcher(new BFMatcher());

	printf("\nLoading dictionary");

	// Loading bag of words from file
	Mat dictionaryF;
	FileStorage fs(ih::MNT_SDCARD_DICTIONARY_SIFT_YML, FileStorage::READ);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	fs["dictionary"] >> dictionaryF;
	fs.release();
	Mat dictionary;
	printf("\nDictionary sizes %d x %d", dictionaryF.cols, dictionaryF.rows);
	dictionaryF.convertTo(dictionary, CV_32F);
	bowDE.setVocabulary(dictionary);

	printf("\nIterating over images ");

	try {
		std::ifstream ifs(dataset_file.c_str());

		std::string path;

		while (std::getline(ifs, path)) {
			Mat matFeature = getSIFTFeatures(TAG, fastDetector, bowDE,
					path.c_str());
			std::ofstream feature_file;
			string output_file = output_dir + SplitFilename(path);
			output_file = output_file.replace(output_file.find("jpg"), 3,
					"bin");
			feature_file.open(output_file.c_str());
			float * floatFeature = (float *) matFeature.data;
			feature_file
					<< printFloatVar(floatFeature, ih::DICTIONARY_SIZE_SIFT);

			matFeature.release();
			feature_file.close();

		}
	} catch (const std::exception & e) {
		printf("\nException %s", e.what());
	}

}

void compute(cv::Mat queryDescriptors, cv::Mat& _imgDescriptor,
		BFMatcher matcher, cv::Mat vocabulary,
		std::vector<std::vector<int> >* pointIdxsOfClusters) {
	CV_Assert(!vocabulary.empty());

	int clusterCount = vocabulary.rows;

	// Match keypoint descriptors to cluster center (to vocabulary)
	std::vector<DMatch> matches;

	matcher.match(queryDescriptors, matches);

	// Compute image descriptor
	if (pointIdxsOfClusters) {
		pointIdxsOfClusters->clear();
		pointIdxsOfClusters->resize(clusterCount);
	}

	_imgDescriptor.create(1, clusterCount, CV_32FC1);
	_imgDescriptor.setTo(Scalar::all(0));

	Mat imgDescriptor = _imgDescriptor;

	float *dptr = imgDescriptor.ptr<float>();
	for (size_t i = 0; i < matches.size(); i++) {
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx; // cluster index
		CV_Assert(queryIdx == (int )i);

		dptr[trainIdx] = dptr[trainIdx] + 1.f;
		if (pointIdxsOfClusters)
			(*pointIdxsOfClusters)[trainIdx].push_back(queryIdx);
	}

	// Normalize image descriptor.
	imgDescriptor /= queryDescriptors.size().height;
}

Mat getHOGFeatures(const char* TAG, BFMatcher matcher,
		FastFeatureDetector detector, HOGDescriptor descriptor,
		cv::Mat vocabulary, const char* path) {
	vector<KeyPoint> keypoints;

	printf("\nRunning HOG on %s", path);
	Mat featureVector = Mat(0, ih::DICTIONARY_SIZE_HOG, CV_32F);
	//extract BoW (or BoF) descriptor from given image
	Mat img = loadScaledImage_desc(path, CV_LOAD_IMAGE_GRAYSCALE);

	Mat allDescriptors(0, 0, CV_32F);

	printf("Image: %s \n", path);
	detector.detect(img, keypoints);
	KeyPointsFilter::removeDuplicated(keypoints);
	KeyPointsFilter::runByImageBorder(keypoints, img.size(), 16);
	KeyPointsFilter::retainBest(keypoints, ih::MAXIMUM_KEYPOINTS);
	for (KeyPoint kp : keypoints) {
		vector<float> descriptors;
		vector<Point> locations;
		Mat imgCut(32, 32, CV_8U);
		int pad = 32 / 2;
		img(Rect(kp.pt.x - pad, kp.pt.y - pad, kp.size, kp.size)).copySize(
				imgCut);
//				descriptor.compute(imgCut, descriptors, Size(0, 0), Size(0, 0), locations);
		descriptor.compute(imgCut, descriptors);
//				std::cout << "Descriptors size: " << descriptors.size() << std::endl;
		Mat dctmat(descriptors, 0);
		allDescriptors.push_back(dctmat);

	}
	img.release();
	std::vector<std::vector<int> > pointIdxsOfClusters;
	compute(allDescriptors, featureVector, matcher, vocabulary,
			&pointIdxsOfClusters);
	return featureVector;
}

void extractlist_HOG_1internal(string dataset_file, string output_dir) {
	const char* TAG = "jni-goldenretrieval";

	HOGDescriptor descriptor(Size(32, 32), Size(8, 8), Size(4, 4), Size(4, 4),
			9);
	FastFeatureDetector detector;

	BFMatcher matcher(cv::NORM_L2);

	printf("\nLoading dictionary");

	// Loading bag of words from file
	Mat dictionaryF;
	FileStorage fs(ih::MNT_SDCARD_DICTIONARY_HOG_YML, FileStorage::READ);
	fs["dictionary"] >> dictionaryF;
	fs.release();
	Mat dictionary;
	printf("\nDictionary sizes %d x %d", dictionaryF.cols, dictionaryF.rows);
	dictionaryF.convertTo(dictionary, 0);
	matcher.add(dictionaryF);

	printf("\nIterating over images ");

	try {
		std::ifstream ifs(dataset_file.c_str());

		std::string path;

		while (std::getline(ifs, path)) {
			Mat matFeature = getHOGFeatures(TAG, matcher, detector, descriptor,
					dictionaryF, path.c_str());
			std::ofstream feature_file;
			string output_file = output_dir + SplitFilename(path);
			output_file = output_file.replace(output_file.find("jpg"), 3,
					"bin");
			feature_file.open(output_file.c_str());
			float * floatFeature = (float *) matFeature.data;
			feature_file
					<< printFloatVar(floatFeature, ih::DICTIONARY_SIZE_SIFT);

			matFeature.release();
			feature_file.close();

		}
	} catch (const std::exception & e) {
		printf("\nException %s", e.what());
	}

}



Mat getLATCHFeatures(const char* TAG, FastFeatureDetector fastDetector,
		features2d::LATCHDescriptorExtractorImpl extractor, BFMatcher matcher, cv::Mat vocabulary, const char* path) {
	vector<KeyPoint> keypoints;
	Mat featureVector = Mat(0, ih::DICTIONARY_SIZE_LATCH, CV_8U);

	Mat img = loadScaledImage_desc(path, CV_LOAD_IMAGE_GRAYSCALE);
	printf("\nRunning detection on %s", path);
	printf("\nImage properties %d x %d", img.cols, img.rows);
	fastDetector.detect(img, keypoints);
	printf("\nKeypoints before %zu", keypoints.size());
	KeyPointsFilter::retainBest(keypoints, ih::MAXIMUM_KEYPOINTS);
	printf("\nKeypoints after %zu", keypoints.size());

	printf("\nRunning LATCH on %s", path);
	Mat matDescriptor;
	//extract BoW (or BoF) descriptor from given image
	extractor.compute(img, keypoints, matDescriptor);
//	matDescriptor.convertTo(matDescriptor, CV_32F);
	img.release();


	std::vector<std::vector<int> > pointIdxsOfClusters;
	compute(matDescriptor, featureVector, matcher, vocabulary,
			&pointIdxsOfClusters);
	return featureVector;
}

void extractlist_LATCH_1internal(string dataset_file, string output_dir) {
	const char* TAG = "jni-goldenretrieval";

	int bytes = 32; bool rotationInvariance = true; int half_ssd_size = 3;
	FastFeatureDetector detector;
	features2d::LATCHDescriptorExtractorImpl extractor(bytes, rotationInvariance, half_ssd_size);;
	vector<KeyPoint> keypoints;
	BFMatcher matcher(cv::NORM_HAMMING);

	printf("\nLoading dictionary");
	 // Loading bag of words from file
	Mat dictionary;
	FileStorage fs(ih::MNT_SDCARD_DICTIONARY_LATCH_YML, FileStorage::READ);
	fs["dictionary"] >> dictionary;
//	Mat dictionary;
	dictionary.convertTo(dictionary, CV_8U);
	printf("\nDictionary sizes %d x %d", dictionary.cols, dictionary.rows);
	matcher.add(std::vector<cv::Mat>(1, dictionary));
	fs.release();
	printf("\nIterating over images ");
	const clock_t begin_time = clock();
	try {
		std::ifstream ifs(dataset_file.c_str());

		std::string path;
		int iter = 1;
		while (std::getline(ifs, path)) {
			Mat matFeature = getLATCHFeatures(TAG, detector, extractor,matcher,dictionary,
					path.c_str());
			std::ofstream feature_file;
			string output_file = output_dir + SplitFilename(path);
			output_file = output_file.replace(output_file.find("jpg"), 3,
					"bin");
			feature_file.open(output_file.c_str());
			float * floatFeature = (float *) matFeature.data;
			feature_file
					<< printFloatVar(floatFeature, ih::DICTIONARY_SIZE_LATCH);

			matFeature.release();
			feature_file.close();
			iter ++;
		}
		std::cout << "Time spent per iter: "<< (float( clock () - begin_time ) /  CLOCKS_PER_SEC) / iter;


	} catch (const std::exception & e) {
		printf("\nException %s", e.what());
	}

}



