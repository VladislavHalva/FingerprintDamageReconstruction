#include "OrientationDiscontinuityDetector.h"
#include "DamageDetector.h"


OrientationDiscontinuityDetector::OrientationDiscontinuityDetector() {
}


cv::Mat OrientationDiscontinuityDetector::detectDiscontinuities(Image* image) {
	cv::Mat orientationField = image->getNonSmoothedOrientationField();
	cv::Mat backgroundMask = image->getBackgroundMask();
	cv::Mat discontinuityMap = cv::Mat::zeros(orientationField.size(), CV_8U);

	//check all blocks for continuous difference of field
	for (int blockX = 0; blockX < discontinuityMap.cols; blockX++) {
		for (int blockY = 0; blockY < discontinuityMap.rows; blockY++) {
			//skip background areas
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				discontinuityMap.at<unsigned char>(blockY, blockX) = BACKGROUND;
				continue;
			}

			double currentBlockOrientation = orientationField.at<double>(blockY, blockX);
		    
		    //row-wise check for discontinuity
			if (blockX != (discontinuityMap.cols - 1)) {
				double nextInRowBlockOrientation = orientationField.at<double>(blockY, blockX + 1);

				if (abs(currentBlockOrientation - nextInRowBlockOrientation) > 45) {
					discontinuityMap.at<unsigned char>(blockY, blockX) = LOW_DAMAGE;
				}
			}

			//column-wise check for discontinuity
			if (blockY != (discontinuityMap.rows - 1)) {
				double nextInColBlockOrientation = orientationField.at<double>(blockY + 1, blockX);

				if (abs(currentBlockOrientation - nextInColBlockOrientation) > 45) {
                    if(discontinuityMap.at<unsigned char>(blockY, blockX) == LOW_DAMAGE)
                    {
						discontinuityMap.at<unsigned char>(blockY, blockX) = DAMAGED;
                    }
					else {
						discontinuityMap.at<unsigned char>(blockY, blockX) = LOW_DAMAGE;
					}
				}
			}
		}
	}

	discontinuityMap = suppressErroneousEstimations(discontinuityMap, image->getBackgroundMask());
    return discontinuityMap;
}


cv::Mat OrientationDiscontinuityDetector::suppressErroneousEstimations(cv::Mat discontinuityMap, cv::Mat backgroundMask) {

	for (int blockX = 0; blockX < discontinuityMap.cols; blockX++) {
		for (int blockY = 0; blockY < discontinuityMap.rows; blockY++) {

			//skip background
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				continue;
			}

			int discontinuitiesInNeighborhood = 0;

			//how many neighborhood blocks is discontinuous(blockX+u,blockY+v)
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					if (Image::isElementInMatSizeRange(blockX + u, blockY + v, discontinuityMap)) {
						int discontinuityClassification = (int)discontinuityMap.at<unsigned char>(blockY + v, blockX + u);
						if (discontinuityClassification == DAMAGED || discontinuityClassification == LOW_DAMAGE) {
							discontinuitiesInNeighborhood++;
						}
					}
				}
			}

			//border elements
			if (Image::isElementBorderElementOfMat(blockX, blockY, discontinuityMap)) {
				if (discontinuitiesInNeighborhood > 3) {
					discontinuityMap.at<unsigned char>(blockY, blockX) = DAMAGED;
				}
			}

			//a lot of damage in neigh --> high probability, that will be damaged either
			else if (discontinuitiesInNeighborhood > 5) {
				discontinuityMap.at<unsigned char>(blockY, blockX) = DAMAGED;
			}

			//too little damage in neigh --> not damaged
			else if (discontinuitiesInNeighborhood < 3) {
				discontinuityMap.at<unsigned char>(blockY, blockX) = NO_DAMAGE;
			}
		}
	}

	return discontinuityMap;
}

