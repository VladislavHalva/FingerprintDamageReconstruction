#pragma once
#include "Filter.h"

#define BANK_SIZE 20

class GaborFilter : public Filter
{
private:
	cv::Size kernelSize = cv::Size(5, 5);
    double stdDev = 4.0;
	double aspectRatio = 0.02;
	double offset = 0;
	vector<double> bankFiltersOrientations;
	vector<double> bankFiltersFrequencies;
	cv::Mat gaborFilterBank[BANK_SIZE][BANK_SIZE];

public:
    GaborFilter();
	void setup(Image* image);
    void createBankOfGaborFilters();
    void filter() override;
    double getMaxInMatWithValidityMask(const cv::Mat& mat, const cv::Mat& validityMask);
    double getMinInMatWithValidityMask(const cv::Mat& mat, const cv::Mat& validityMask);
	int getClosestValueIndex(double value, const vector<double>& vector);
};



//  ksize = size of the Gabor kernel
//	sigma = standard deviation of the Gaussian function
//	theta = field of the normal to the parallel stripes
//	lambda = wavelength of the sinusoidal factor
//	gamma = spatial aspect ratio --> ellipticity (0.5 default, 1 circle)
//	psi = phase offset.
