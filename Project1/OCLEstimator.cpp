#include "OCLEstimator.h"
#include "OrientationsEstimator.h"
#include "BackgroundSubstractor.h"


OCLEstimator::OCLEstimator() {
}



/**
 * algorithm from E. Lim - Fingerprint quality and validity analysis
 */
cv::Mat OCLEstimator::computeOcl(Image* image)
{
	cv::Mat img = image->getProcessedImage();
    cv::Mat orientationField = image->getNonSmoothedOrientationField();
	cv::Mat backgroundMask = image->getBackgroundMask();
	int blockSize = image->getBlockSize();

	cv::Mat gradX = OrientationsEstimator::calcGradX(img, CV_64F);
	cv::Mat gradY = OrientationsEstimator::calcGradY(img, CV_64F);
    
    cv::Mat oclMap = cv::Mat::zeros(orientationField.size(), CV_64F);

    //compute field certainty level for each block
    for(int blockX = 0; blockX < oclMap.cols; blockX++)
    {
        for(int blockY = 0; blockY < oclMap.rows; blockY++)
        {
            //skip background
			if(BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask))
			{
				oclMap.at<double>(blockY, blockX) = BACKGROUND;
				continue;
			}
            
            double covariance[3] = { 0, 0, 0 };

            //all pixels belonging to block
            for(int i = 0; i < blockSize; i++)
            {
                for(int j = 0; j < blockSize; j++)
                {
					int pixelX = blockX * blockSize + i;
					int pixelY = blockY * blockSize + j;
					double pixelGradX = gradX.at<double>(pixelY, pixelX);
					double pixelGradY = gradY.at<double>(pixelY, pixelX);

					covariance[0] += pixelGradX * pixelGradX;
					covariance[1] += pixelGradY * pixelGradY;
					covariance[2] += pixelGradX * pixelGradY;
                }
            }

			covariance[0] = covariance[0] / (blockSize * blockSize);
			covariance[1] = covariance[1] / (blockSize * blockSize);
			covariance[2] = covariance[2] / (blockSize * blockSize);

			double lambdaMin = calcLambdaMin(covariance);
			double lambdaMax = calcLambdaMax(covariance);			

            if(lambdaMax != 0) oclMap.at<double>(blockY, blockX) = lambdaMin / lambdaMax;
        }
    }

    //convert to range 0 - 255
	cv::Mat oclMapNorm;
	oclMap.convertTo(oclMapNorm, CV_8U, 255.0);

	reduceErrorEstimations(oclMapNorm);
	
    return oclMapNorm;
}


double OCLEstimator::calcLambdaMin(double covariance[3]) {
	return ((covariance[0] + covariance[1]) -
		sqrt(pow(covariance[0] - covariance[1], 2) + 4 * pow(covariance[2], 2))) / 2.;
}


double OCLEstimator::calcLambdaMax(double covariance[3]) {
	return ((covariance[0] + covariance[1]) +
		sqrt(pow(covariance[0] - covariance[1], 2) + 4 * pow(covariance[2], 2))) / 2.;
}

void OCLEstimator::reduceErrorEstimations(cv::Mat oclMap)
{
	for (int blockX = 0; blockX < oclMap.cols; blockX++) {
		for (int blockY = 0; blockY < oclMap.rows; blockY++) {

			int lowOCLInNeighborhood = 0;

			//how many neighborhood blocks have low ocl
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					if (Image::isElementInMatSizeRange(blockX + u, blockY + v, oclMap)) {
						if ((int)oclMap.at<unsigned char>(blockY + v, blockX + u) > 10 && 
							(int)oclMap.at<unsigned char>(blockY + v, blockX + u) != 255) {
							lowOCLInNeighborhood++;
						}
					}
				}
			}

			//a lot of high quality in neigh --> high q
			if (lowOCLInNeighborhood < 3) {
				oclMap.at<unsigned char>(blockY, blockX) = 0;
			}
		}
	}
}
