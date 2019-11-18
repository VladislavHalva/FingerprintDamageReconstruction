#pragma once
#include <opencv2/core/types.hpp>

class BasicOperations
{
public:
	BasicOperations();
	static double DegToRad(double deg);
	static double RadToDeg(double rad);
	static std::vector<cv::Point_<int>> getSurroundingPoints(int pixelX, int pixelY);
};

