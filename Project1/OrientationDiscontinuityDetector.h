#pragma once
#include "DamageDetector.h"

class OrientationDiscontinuityDetector {
public:
	OrientationDiscontinuityDetector();
    cv::Mat detectDiscontinuities(Image* image);
	cv::Mat suppressErroneousEstimations(cv::Mat discontinuityMap, cv::Mat backgroundMask);
};

