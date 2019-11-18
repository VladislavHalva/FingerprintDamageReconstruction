#include "BasicOperations.h"
#include <opencv2/core/cvdef.h>
#include <vector>
#include <opencv2/core/mat.hpp>

using namespace std;

BasicOperations::BasicOperations()
{
}

double BasicOperations::DegToRad(double deg)
{
	return deg * CV_PI / 180.;
}

double BasicOperations::RadToDeg(double rad)
{
	return rad * 180. / CV_PI;
}


vector<cv::Point> BasicOperations::getSurroundingPoints(int pixelX, int pixelY)
{
	vector<cv::Point> points;

	points.push_back(cv::Point(pixelX, pixelY + 1));
	points.push_back(cv::Point(pixelX - 1, pixelY + 1));
	points.push_back(cv::Point(pixelX - 1, pixelY));
	points.push_back(cv::Point(pixelX - 1, pixelY - 1));
	points.push_back(cv::Point(pixelX, pixelY - 1));
	points.push_back(cv::Point(pixelX + 1, pixelY - 1));
	points.push_back(cv::Point(pixelX + 1, pixelY));
	points.push_back(cv::Point(pixelX + 1, pixelY + 1));

	return points;
}

