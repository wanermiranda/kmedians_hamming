#include "imgHelpers.hpp"


using namespace cv;
using namespace std;


Mat loadScaledImageSqrd(const char *path, int flags,
		float target_width) {
	Mat img = imread(path, flags);

	int width = img.cols, height = img.rows;

	cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

	int max_dim = (width >= height) ? width : height;
	float scale = ((float) target_width) / max_dim;
	cv::Rect roi;
	if (width >= height) {
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	} else {
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}

	cv::resize(img, square(roi), roi.size());

	return square;

}

Mat loadImage(const char *path, int flags) {
	Mat img = imread(path, flags);
	return img;
}
Mat loadScaledImage(const char *path, int flags, float maxSize) {
	Mat img_original = imread(path, flags);

	Mat img;

	int originalSize =
			(img_original.size().height > img_original.size().width) ?
					img_original.size().height : img_original.size().width;

	double ratio = (originalSize > maxSize) ? (maxSize / originalSize) : 1.0;

	resize(img_original, img, Size(), ratio, ratio, INTER_LINEAR);
	img_original.release();

	return img;
}

int getMinorDimension(cv::Mat img_original) {
	return (img_original.size().height < img_original.size().width) ?
					img_original.size().height : img_original.size().width;
}


