#include "ImageArea.h"


ImageArea::ImageArea(vector<cv::Point> points, int pointsState)
{
	this->points = points;
	this->pointsState = pointsState;
}

vector<cv::Point> ImageArea::getPoints()
{
	return this->points;
}

int ImageArea::getPointsState()
{
	return this->pointsState;
}

int ImageArea::getHeight()
{
	if (!this->getPoints().empty()) {
		int minY = this->getPoints().at(0).y;
		int maxY = this->getPoints().at(0).y;

		for (int i = 1; i < this->getPoints().size(); i++) {
			if (this->getPoints().at(i).y < minY) {
				minY = this->getPoints().at(i).y;
			}
			if (this->getPoints().at(i).y > maxY) {
				maxY = this->getPoints().at(i).y;
			}
		}

		return maxY - minY;
	}
	else {
		return 0;
	}
}

int ImageArea::getWidth()
{
	if (!this->getPoints().empty())
	{
		int minX = this->getPoints().at(0).x;
		int maxX = this->getPoints().at(0).x;

		for (int i = 1; i < this->getPoints().size(); i++)
		{
			if (this->getPoints().at(i).x < minX)
			{
				minX = this->getPoints().at(i).x;
			}
			if (this->getPoints().at(i).x > maxX)
			{
				maxX = this->getPoints().at(i).x;
			}
		}

		return maxX - minX;
	}
	else
	{
		return 0;
	}
}

int ImageArea::getPointsNumber()
{
	return this->getPoints().size();
}

void ImageArea::addPoint(cv::Point point)
{
	this->points.push_back(point);
}

void ImageArea::setPointsState(int state)
{
	this->pointsState = state;
}



