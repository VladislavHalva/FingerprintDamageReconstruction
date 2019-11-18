#include "Preprocessor.h"


Preprocessor::Preprocessor()
{
}


void Preprocessor::normalize(Image* image)
{
    cv::Mat img = image->getProcessedImage();

    cv::Scalar meanSc, devSc;
    cv::Mat normalizedImage;

    img.copyTo(normalizedImage);

    //calculace mean value and variance
    meanStdDev(img, meanSc, devSc);
    double mean = meanSc.val[0];
    double stdDeviation = devSc.val[0];
    double variance = pow(stdDeviation, 2);

    //set desired values
    double desiredMean = 100;
    double desiredVariance = 1000;

    //normalize
    for (int coordX = 0; coordX < img.cols; coordX++)
    {
        for (int coordY = 0; coordY < img.rows; coordY++)
        {
            int currentPixel = (int)img.at<unsigned char>(coordY, coordX);

            if ((int)img.at<unsigned char>(coordY, coordX) > mean)
            {
                normalizedImage.at<unsigned char>(coordY, coordX) = static_cast<unsigned char>(
                    desiredMean + sqrt(desiredVariance * pow(currentPixel - mean, 2) / variance));
            }
            else
            {
                normalizedImage.at<unsigned char>(coordY, coordX) = static_cast<unsigned char>(
                    desiredMean - sqrt(desiredVariance * pow(currentPixel - mean, 2) / variance));
            }
        }
    }
    image->setProcessedImage(normalizedImage);
}


void Preprocessor::smoothenImage(Image* image, int sigma)
{
    cv::Mat img = image->getProcessedImage();
    GaussianBlur(img, img, cv::Size(3, 3), sigma, sigma, cv::BORDER_DEFAULT);
    image->setProcessedImage(img);
}


void Preprocessor::equalize(Image* image)
{
    double minD, maxD;
    minMaxLoc(image->getProcessedImage(), &minD, &maxD);
    int min = static_cast<int>(minD);
    int max = static_cast<int>(maxD);

    cv::Mat equalized;
    image->getProcessedImage().convertTo(equalized, CV_8U, 255.0 / (max - min), min * 255.0 / (min - max));
    image->setProcessedImage(equalized);
}


cv::Mat Preprocessor::binarize(Image* image) 
{
	cv::Mat img = image->getProcessedImage();
	cv::Mat processedImg = cv::Mat::zeros(img.size(), CV_8U);
	int blockSize = image->getBlockSize();

	cv::adaptiveThreshold(img, processedImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 0);

	return processedImg;
}

