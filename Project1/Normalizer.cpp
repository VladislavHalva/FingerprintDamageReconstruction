#include "Preprocessor.h"



Preprocessor::Preprocessor() {
}


void Preprocessor::normalize(Image* image) {
	
	cv::Mat img = image->getImage();

	cv::Scalar meanSc, devSc;
	double mean, stdDeviation, variance;
	double desiredMean, desiredVariance;
	cv::Mat normalizedImage;

	img.copyTo(normalizedImage);

	//calculace mean value and variance
	cv::meanStdDev(img, meanSc, devSc);
	mean = meanSc.val[0];
	stdDeviation = devSc.val[0];
	variance = pow(stdDeviation, 2);

	//set desired values
	desiredMean = 100;
	desiredVariance = 1000;

	//normalize
	for (int coordX = 0; coordX < img.cols; coordX++) {
		for (int coordY = 0; coordY < img.rows; coordY++) {
			int currentPixel = (int)img.at<unsigned char>(coordY, coordX);

			if ((int)img.at<unsigned char>(coordY, coordX) > mean) {
				normalizedImage.at<unsigned char>(coordY, coordX) = static_cast<unsigned char>(
					desiredMean + sqrt(desiredVariance*pow(currentPixel - mean, 2) / variance));
			}
			else {
				normalizedImage.at<unsigned char>(coordY, coordX) = static_cast<unsigned char>(
					desiredMean - sqrt(desiredVariance*pow(currentPixel - mean, 2) / variance));
			}
		}
	}
	image->updateImage(normalizedImage);
}



void Preprocessor::smoothenImage(Image* image, int sigma) {
	cv::Mat img = image->getImage();
	cv::GaussianBlur(img, img, cv::Size(3, 3), sigma, sigma, cv::BORDER_DEFAULT);
	image->updateImage(img);
}


