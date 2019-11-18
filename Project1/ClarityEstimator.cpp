#include "ClarityEstimator.h"
#include "BackgroundSubstractor.h"


ClarityEstimator::ClarityEstimator() {
}

cv::Mat ClarityEstimator::computeClarity(Image* image)
{
	cv::Mat img = image->getProcessedImage();
	cv::Mat backgroundMask = image->getBackgroundMask();
	int blockSize = image->getBlockSize();

	cv::Mat clarityMap = cv::Mat::zeros(img.rows / blockSize, img.cols / blockSize, CV_8U);

	for (int blockX = 0; blockX < clarityMap.cols; blockX++) {
		for (int blockY = 0; blockY < clarityMap.rows; blockY++) {
			//skip background areas
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				clarityMap.at<unsigned char>(blockY, blockX) = BACKGROUND;
				continue;
			}

			//calculate mean and variance in block
			cv::Scalar meanSc, devSc;
			double mean, stdDev, variance;

			//subduct only the current block
			cv::Mat block = img(cv::Rect(blockX * blockSize, blockY * blockSize, blockSize, blockSize));
			meanStdDev(block, meanSc, devSc);
			mean = meanSc.val[0];
			stdDev = devSc.val[0];
			variance = pow(stdDev, 2);

            //low quality areas will have lower mean value of gray intensity
            if(mean < 100)
            {
				clarityMap.at<unsigned char>(blockY, blockX) = LOW_CLARITY;
            }
			else
			{
				clarityMap.at<unsigned char>(blockY, blockX) = HIGH_CLARITY;
			}

			//low quality areas will have lower variance value of gray intensity
			if (variance < 200) {
				clarityMap.at<unsigned char>(blockY, blockX) = LOW_CLARITY;
			}
			else {
				clarityMap.at<unsigned char>(blockY, blockX) = HIGH_CLARITY;
			}

		}
	}

	clarityMap = suppressErroneousEstimations(clarityMap, backgroundMask);
	return clarityMap;
}


cv::Mat ClarityEstimator::suppressErroneousEstimations(cv::Mat clarityMap, cv::Mat backgroundMask)
{
	for (int blockX = 0; blockX < clarityMap.cols; blockX++) {
		for (int blockY = 0; blockY < clarityMap.rows; blockY++) {
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				continue;
			}

			int lowClarityInNeighborhood = 0;

			//how many neighborhood blocks have bad clarity
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					if (Image::isElementInMatSizeRange(blockX + u, blockY + v, clarityMap)) {
						if ((int)clarityMap.at<unsigned char>(blockY + v, blockX + u) == LOW_CLARITY) {
							lowClarityInNeighborhood++;
						}
					}
				}
			}

			//a lot of low quality in neigh --> low q
			if (lowClarityInNeighborhood > 5) {
				clarityMap.at<unsigned char>(blockY, blockX) = LOW_CLARITY;
			}

			//a lot of high quality in neigh --> high q
			else if (lowClarityInNeighborhood < 3) {
				clarityMap.at<unsigned char>(blockY, blockX) = HIGH_CLARITY;
			}
		}
	}

	return clarityMap;
}

