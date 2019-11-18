#include "RidgeClarityEstimator.h"
#include "BackgroundSubstractor.h"
#include "Preprocessor.h"


RidgeClarityEstimator::RidgeClarityEstimator() {
}


cv::Mat RidgeClarityEstimator::computeRidgeClarity(Image* image)
{
	cv::Mat binarizedImg = Preprocessor::binarize(image);
	cv::Mat orientationField = image->getOrientationField();
	cv::Mat backgroundMask = image->getBackgroundMask();
    int blockSize = image->getBlockSize();
	int windowWidth = image->getWindowWidth();

	cv::Mat ridgeClarityMap = cv::Mat::zeros(binarizedImg.rows / blockSize, binarizedImg.cols / blockSize, CV_8U);

	for (int blockX = 0; blockX < ridgeClarityMap.cols; blockX++) {
		for (int blockY = 0; blockY < ridgeClarityMap.rows; blockY++) {
			//skip background areas
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				ridgeClarityMap.at<unsigned char>(blockY, blockX) = BACKGROUND;
				continue;
			}

			vector<double> avgIntensitiesInLine(windowWidth, 0.0);

			int blockCenterPixelX = static_cast<int>(blockX * blockSize + (blockSize - 1) / 2);
			int blockCenterPixelY = static_cast<int>(blockY * blockSize + (blockSize - 1) / 2);

			double orientationRad = image->getOrientationField().at<double>(blockY, blockX) * CV_PI / 180.0;

			double cosine = cos(orientationRad);
			double sine = sin(orientationRad);

			for (int lineIndex = 0; lineIndex < windowWidth; lineIndex++) {
				//pixels withing the image area                
				int validValuesOfIntensity = 0;

				//iterate over pixels in one line according to the window field
				for (int linePixelIndex = 0; linePixelIndex < blockSize; linePixelIndex++) {
					//calculating coordinates of next pixel in line according to ridge line field 
					int linePixelX = static_cast<int>(
						blockCenterPixelX + (linePixelIndex - blockSize / 2) * cosine +
						(lineIndex - windowWidth / 2) * sine);
					int linePixelY = static_cast<int>(
						blockCenterPixelY + (linePixelIndex - blockSize / 2) * sine +
						(windowWidth / 2 - lineIndex) * cosine);

					//adding value to Sum of intensities of pixels in line
					if (Image::isElementInMatSizeRange(linePixelX, linePixelY, binarizedImg)) {
						//add only if pixel belongs to the processedImage area
						avgIntensitiesInLine[lineIndex] += (double)binarizedImg.at<unsigned char>(linePixelY, linePixelX);
						validValuesOfIntensity++;
					}
				}
				//finishing calculation of one particular line average intensity	
				if (validValuesOfIntensity != 0) {
					avgIntensitiesInLine[lineIndex] = avgIntensitiesInLine[lineIndex] / validValuesOfIntensity;
				}
			}

            //how many lines in one block have stable value of gray intensity = low variance
			int goodClarityLinesPerBlock = windowWidth;
            for(int index = 0; index < avgIntensitiesInLine.size(); index++)
            {
                if(avgIntensitiesInLine.at(index) > 50 && avgIntensitiesInLine.at(index) < 200)
                {
					goodClarityLinesPerBlock--;
                }
            }

            //at least half of lines per block have to have good clarity --> low variance in intensity
            if(goodClarityLinesPerBlock > windowWidth * 0.5)
            {
				ridgeClarityMap.at<unsigned char>(blockY, blockX) = HIGH_CLARITY;
            }
			else
			{
				ridgeClarityMap.at<unsigned char>(blockY, blockX) = LOW_CLARITY;
			}
		}
	}

    ridgeClarityMap = suppressErroneousEstimations(ridgeClarityMap, backgroundMask);
	return ridgeClarityMap;
}


cv::Mat RidgeClarityEstimator::suppressErroneousEstimations(cv::Mat clarityIndexMap, cv::Mat backgroundMask)
{
    for(int blockX = 0; blockX < clarityIndexMap.cols; blockX++)
    {
        for(int blockY = 0; blockY < clarityIndexMap.rows; blockY++)
        {
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				continue;
			}

			int lowClarityInNeighborhood = 0;

			//how many neighborhood blocks have bad clarity
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					if (Image::isElementInMatSizeRange(blockX + u, blockY + v, clarityIndexMap)) {
						if ((int)clarityIndexMap.at<unsigned char>(blockY + v, blockX + u) == LOW_CLARITY) {
							lowClarityInNeighborhood++;
						}
					}
				}
			}

            //a lot of low quality in neigh --> low q
			if (lowClarityInNeighborhood > 6) 
			{
				clarityIndexMap.at<unsigned char>(blockY, blockX) = LOW_CLARITY;
			}

            //a lot of high quality in neigh --> high q
			else if (lowClarityInNeighborhood < 3) 
			{
				clarityIndexMap.at<unsigned char>(blockY, blockX) = HIGH_CLARITY;
			}
        }
    }

	return clarityIndexMap;
}
