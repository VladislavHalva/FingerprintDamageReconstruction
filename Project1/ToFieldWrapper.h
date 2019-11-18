#pragma once
#include <opencv2/core/mat.hpp>

typedef struct oFieldWrapper {
	int rangeBegin;
	int rangeEnd;
	int blockSize;
	cv::Mat field;
	cv::Mat thetaX;
	cv::Mat thetaY;
}ToFieldWrapper;
