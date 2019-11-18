#pragma once
#include <opencv2/core/mat.hpp>

using namespace std;

class ImageArea {
private:
	vector<cv::Point> points;
	int pointsState;

public:
	ImageArea(vector<cv::Point> points, int pointsState);

	vector<cv::Point> getPoints();
	int getPointsState();
	int getHeight();
	int getWidth();
	int getPointsNumber();

	void addPoint(cv::Point point);
	void setPointsState(int state);
}; 

