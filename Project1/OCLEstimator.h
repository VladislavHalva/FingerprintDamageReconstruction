#pragma once
#include <opencv2/core/mat.hpp>
#include "Image.h"

class OCLEstimator {
public:
	OCLEstimator();
    double calcLambdaMin(double covariance[3]);
    double calcLambdaMax(double covariance[3]);
    void reduceErrorEstimations(cv::Mat oclMap);
	cv::Mat computeOcl(Image* image);
};

