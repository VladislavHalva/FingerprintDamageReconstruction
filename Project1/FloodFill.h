#pragma once
#include <opencv2/core/mat.hpp>

class FloodFill {
public:
	FloodFill();
	static void floodFillStep(int x, int y, cv::Mat* mat, int oldValue, int newValue, std::vector<cv::Point>* areaBlocks);
	static bool searchFloodFillStep(int x, int y, cv::Mat* mat, int oldValue, int newValue, int searchedValue, std::vector<cv::Point>* allNeighboring);
};

