#pragma once
#include "Image.h"

class RidgeClarityEstimator {
public:
	RidgeClarityEstimator();
	cv::Mat computeRidgeClarity(Image* image);
    cv::Mat suppressErroneousEstimations(cv::Mat clarityIndexMap, cv::Mat backgroundMask);
};

