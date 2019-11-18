#pragma once
#include "Image.h"


class SingularityDetector
{
public:
	SingularityDetector();
	void findSingularities(Image* image);
	void estimatePoincareIndex(Image* image);
	cv::Mat markCoresAndDeltas(const cv::Mat& poincareIndexMap);
	void eliminateFalseCoresAndDeltas(cv::Mat* coresAndDeltas, const cv::Mat& backgroundMask);
	vector<cv::Point> getSurroundingPointsInDefinedOrder(int pixelX, int pixelY);
	vector<double> getOrientationsAtPointsInRadians(const vector<cv::Point_<int>>& points, const cv::Mat orientationsMap);
	void markDamageAreasThatContainCoreOrDelta(Image* image);
	
	static cv::Mat drawSingularityMap(Image* image);
};

