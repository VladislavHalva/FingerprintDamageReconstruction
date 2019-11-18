#pragma once
#include "Image.h"

class ClarityEstimator {
public:
	ClarityEstimator();
	cv::Mat computeClarity(Image* image);
    cv::Mat suppressErroneousEstimations(cv::Mat clarityMap, cv::Mat backgroundMask);
};

