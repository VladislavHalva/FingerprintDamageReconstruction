#include "SingularityDetector.h"
#include "BackgroundSubstractor.h"
#include "BasicOperations.h"
#include "SingularityType.h"


SingularityDetector::SingularityDetector()
{
}

void SingularityDetector::findSingularities(Image* image)
{
	estimatePoincareIndex(image);
}


void SingularityDetector::estimatePoincareIndex(Image* image)
{
	int blockSize = image->getBlockSize();
	cv::Mat orientationField = image->getOrientationField();
	cv::Mat poincareIndexMap = cv::Mat::zeros(orientationField.size(), CV_64F);

	for (int blockX = 0; blockX < orientationField.cols; blockX++)
	{
		for (int blockY = 0; blockY < orientationField.rows; blockY++)
		{
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, image->getBackgroundMask()))
				continue;
			if (Image::isElementBorderElementOfMat(blockX, blockY, orientationField))
				continue;
			if(BackgroundSubstractor::hasBackgroundNeighbor(blockX, blockY, image->getBackgroundMask()))
			{
				continue;
			}

			vector<cv::Point> surroundingPoints = getSurroundingPointsInDefinedOrder(blockX, blockY);
			vector<double> surroundingOrientations = getOrientationsAtPointsInRadians(surroundingPoints, orientationField);

			double sumBeta = 0.;

			for (int i = 0; i < surroundingOrientations.size(); i++)
			{
				double beta;
				double orientationChange = abs(surroundingOrientations.at(i) - surroundingOrientations.at((i + 1) % 8));

				if (orientationChange <= -(CV_PI / 2.))
					beta = orientationChange + CV_PI;
				else if (orientationChange > (-CV_PI / 2.) && orientationChange <= (CV_PI / 2.))
					beta = orientationChange;
				else
					beta = orientationChange - CV_PI;

				sumBeta += beta;
			}

			double poincareIndex = 1 / CV_PI * sumBeta;
			poincareIndexMap.at<double>(blockY, blockX) = poincareIndex;
		}
	}

	cv::Mat coresAndDeltas = markCoresAndDeltas(poincareIndexMap);
	
	eliminateFalseCoresAndDeltas(&coresAndDeltas, image->getBackgroundMask());

	image->setSingularityMap(coresAndDeltas);
}


cv::Mat SingularityDetector::markCoresAndDeltas(const cv::Mat& poincareIndexMap)
{
	cv::Mat coresAndDeltas = cv::Mat::zeros(poincareIndexMap.size(), CV_32S);

	for (int i = 0; i < coresAndDeltas.cols; i++)
	{
		for (int j = 0; j < coresAndDeltas.rows; j++)
		{
			if (poincareIndexMap.at<double>(j, i) > -1 &&
				poincareIndexMap.at<double>(j, i) < -0.5)
			{
				coresAndDeltas.at<int>(j, i) = CORE_OR_DELTA;
			}
			if (poincareIndexMap.at<double>(j, i) > 0.5 &&
				poincareIndexMap.at<double>(j, i) < 1)
			{
				coresAndDeltas.at<int>(j, i) = CORE_OR_DELTA;
			}
		}
	}

	return coresAndDeltas;
}


void SingularityDetector::eliminateFalseCoresAndDeltas(cv::Mat* coresAndDeltas, const cv::Mat& backgroundMask)
{		
	for(int blockX = 0; blockX < coresAndDeltas->cols; blockX++)
	{
		for(int blockY = 0; blockY < coresAndDeltas->rows; blockY++)
		{
			if (Image::isElementBorderElementOfMat(blockX, blockY, backgroundMask))
				continue;
			if(BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask))
				continue;

			vector<cv::Point> surroundingPoints = getSurroundingPointsInDefinedOrder(blockX, blockY);
			surroundingPoints.push_back(cv::Point(blockX, blockY));
			
			vector<cv::Point> singularityPointsInNeigh;
			int cores = 0;
			int deltas = 0;

			//count singularity blocks in neigh
			for (cv::Point point : surroundingPoints)
			{
				if (coresAndDeltas->at<int>(point.y, point.x) == CORE_OR_DELTA) {
					singularityPointsInNeigh.push_back(point);
					cores++;
				}
				else if (coresAndDeltas->at<int>(point.y, point.x) == CORE_OR_DELTA) {
					singularityPointsInNeigh.push_back(point);
					deltas++;
				}
			}

			//lot of singularity points in neigh >> belongs to singularity
			if (singularityPointsInNeigh.size() > 3)
			{
				if (cores > deltas)
					coresAndDeltas->at<int>(blockY, blockX) = CORE_OR_DELTA;
				else
					coresAndDeltas->at<int>(blockY, blockX) = CORE_OR_DELTA;
			}
			//little singularity blocks in neight >> is not singularity 
			else
			{
				coresAndDeltas->at<int>(blockY, blockX) = 0;
			}
		}
	}
}


vector<cv::Point> SingularityDetector::getSurroundingPointsInDefinedOrder(int pixelX, int pixelY)
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


vector<double> SingularityDetector::getOrientationsAtPointsInRadians(const vector<cv::Point_<int>>& points,
	const cv::Mat orientationsMap)
{
	vector<double> orientations;
	for (cv::Point point : points)
	{
		orientations.push_back(
			BasicOperations::DegToRad(
				orientationsMap.at<double>(point.y, point.x)));
	}
	return orientations;
}

void SingularityDetector::markDamageAreasThatContainCoreOrDelta(Image* image)
{
	vector<ImageArea> areas = image->getHighlyDamagedAreas();
	cv::Mat singularityMap = image->getSingularityMap();
	if (singularityMap.empty()) return;

	for (ImageArea area : areas)
	{
		bool containsSingularity = false;

		for (cv::Point block : area.getPoints())
		{
			if (singularityMap.at<int>(block.y, block.x) == CORE_OR_DELTA) {
				containsSingularity = true;
				break;
			}
		}

		area.setPointsState(SINGULARITY);
	}
}

cv::Mat SingularityDetector::drawSingularityMap(Image* image)
{
	cv::Mat map = image->getSingularityMap();

	cv::Mat show = Image::convertToUCharAndExtendToRange0_255(map);
	show = Image::extendBlocksToFullSizeImage(show, image->getBlockSize(), image->getSize());
	return show;
}

