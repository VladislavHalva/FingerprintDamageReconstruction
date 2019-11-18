#pragma once
#include <opencv2/core/mat.hpp>

typedef struct ToFieldWrapper {
	int rangeBegin;
	int rangeEnd;
	int blockSize;
	cv::Mat field;
}ToFieldWrapper;
