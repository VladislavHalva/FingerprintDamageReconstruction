#include "FrequencyEstimator.h"
#include "Filter.h"
#include <experimental/filesystem>
#include "BackgroundSubstractor.h"
#include "OrientationsEstimator.h"


FrequencyEstimator::FrequencyEstimator()
{
}


void FrequencyEstimator::computeFrequencyField(Image* image)
{
    cv::Mat img = image->getProcessedImage();
    int blockSize = image->getBlockSize();
    int windowWidth = image->getWindowWidth();

    cv::Mat frequencyField(img.rows / blockSize, img.cols / blockSize, CV_64F);

    //iterate over blocks of frequency field (i,j)
    for (int i = 0; i < frequencyField.cols; i++)
    {
        for (int j = 0; j < frequencyField.rows; j++)
        {
            if (BackgroundSubstractor::isBackgroundBlock(i, j, image->getBackgroundMask()))
            {
                frequencyField.at<double>(j, i) = -1;
                continue;
            }

            vector<double> xSignature(windowWidth, 0.0);

            int blockCenterPixelX = static_cast<int>(i * blockSize + (blockSize - 1) / 2);
            int blockCenterPixelY = static_cast<int>(j * blockSize + (blockSize - 1) / 2);

            double orientationRad = image->getOrientationField().at<double>(j, i) * CV_PI / 180.0;

            double cosine = cos(orientationRad);
            double sine = sin(orientationRad);

            //calculate all xSignature for current block (window)
            for (int xSignIndex = 0; xSignIndex < windowWidth; xSignIndex++)
            {
                //pixels withing the image area                
                int validValuesOfIntensity = 0;

                //iterate over pixels in one line according to the window field
                for (int linePixelIndex = 0; linePixelIndex < blockSize; linePixelIndex++)
                {
                    //calculating coordinates of next pixel in line according to ridge line field 
                    int linePixelX = static_cast<int>(
                        blockCenterPixelX + (linePixelIndex - blockSize / 2) * cosine +
                        (xSignIndex - windowWidth / 2) * sine);
                    int linePixelY = static_cast<int>(
                        blockCenterPixelY + (linePixelIndex - blockSize / 2) * sine +
                        (windowWidth / 2 - xSignIndex) * cosine);

                    //adding value to Sum of intensities of pixels in line
                    if (Image::isElementInMatSizeRange(linePixelX, linePixelY, img))
                    {
                        //add only if pixel belongs to the processedImage area
                        xSignature[xSignIndex] += (double)img.at<unsigned char>(linePixelY, linePixelX);
                        validValuesOfIntensity++;
                    }
                }
                //finishing calculation of one particular x-signature (average intensity)	
                if (validValuesOfIntensity != 0)
                {
                    xSignature[xSignIndex] = xSignature[xSignIndex] / validValuesOfIntensity;
                }
            }
			
			int kernelSize = ((blockSize / 9 * 7) >= 3) ? (blockSize / 9 * 7) : 3;
            xSignature = smoothenSignatures(xSignature, kernelSize, 1);

            vector<int> locMaxIndexes = findLocalMax(xSignature);
            vector<int> locMinIndexes = findLocalMin(xSignature);

            bool frequencyInMinimums = frequencyFound(locMinIndexes, windowWidth);
            bool frequencyInMaximums = frequencyFound(locMaxIndexes, windowWidth);

            if (frequencyInMaximums || frequencyInMinimums)
            {
                //frequency found
                int period;
                if (frequencyInMaximums && frequencyInMinimums)
                {
                    //freq in minimums and maximums --> average
                    period = (getPeriodLenght(locMaxIndexes) + getPeriodLenght(locMinIndexes)) / 2;
                }
                else if (frequencyInMaximums)
                {
                    //freq in maximums only
                    period = getPeriodLenght(locMaxIndexes);
                }
                else
                {
                    //freq in minimums only
                    period = getPeriodLenght(locMinIndexes);
                }

                frequencyField.at<double>(j, i) = 1.0 / period;
            }
            else
            {
                //not frequency could be estimated
                frequencyField.at<double>(j, i) = -1;
            }
        }
    }

    interpolateFreqField(img, frequencyField, blockSize, image->getBackgroundMask());
    image->setFrequencyField(frequencyField);
}


void FrequencyEstimator::interpolateFreqField(cv::Mat img, cv::Mat frequencyField, int blockSize,
                                              cv::Mat backgroundMask)
{
    cv::Mat enhancedFrequencyField;
    frequencyField.copyTo(enhancedFrequencyField);

    //size should be odd
    int kernelSize = static_cast<int>(blockSize * 1.5);
    (kernelSize % 2 == 0) ? kernelSize-- : kernelSize;

    //create Gaussian kernel
    cv::Mat gaussKernel = Filter::get2DGaussianKernel(kernelSize, kernelSize, 1, 1);


    for (int blockX = 0; blockX < frequencyField.cols; blockX++)
    {
        for (int blockY = 0; blockY < frequencyField.rows; blockY++)
        {
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				//skip background
			    continue;
			}

			//interpolate frequency value if could not be estimated for this block
            if (frequencyField.at<double>(blockY, blockX) == -1)
            {
                int blockCenterX = static_cast<int>(blockX * blockSize + (blockSize - 1) / 2);
                int blockCenterY = static_cast<int>(blockY * blockSize + (blockSize - 1) / 2);

                double numerator = 0.0;
                double denumerator = 0.0;

                for (int kernelX = -kernelSize / 2; kernelX <= kernelSize / 2; kernelX++)
                {
                    for (int kernelY = -kernelSize / 2; kernelY <= kernelSize / 2; kernelY++)
                    {
                        int currentPixelX = blockCenterX + kernelX;
                        int currentPixelY = blockCenterY + kernelY;

                        if (Image::isElementInMatSizeRange(currentPixelX, currentPixelY, img))
                        {
                            int pixelsBlockX = currentPixelX / blockSize;
                            int pixelsBlockY = currentPixelY / blockSize;

                            if (!BackgroundSubstractor::isBackgroundBlock(pixelsBlockX, pixelsBlockY, backgroundMask))
                            {
                                double currentPixelFreq = frequencyField.at<double>(pixelsBlockY, pixelsBlockX);
                                double mu;
                                double delta;
                                (currentPixelFreq <= 0) ? mu = 0 : mu = currentPixelFreq;
                                (currentPixelFreq + 1 <= 0) ? delta = 0 : delta = 1;

                                numerator += gaussKernel.at<double>(kernelY + kernelSize / 2, kernelX + kernelSize / 2) * mu;
                                denumerator += gaussKernel.at<double>(kernelY + kernelSize / 2, kernelX + kernelSize / 2) * delta;
                            }
                        }
                    }
                }
                if (denumerator != 0)
                {
                    enhancedFrequencyField.at<double>(blockY, blockX) = numerator / denumerator;
                }
            }
        }
    }

    enhancedFrequencyField.copyTo(frequencyField);
}


vector<int> FrequencyEstimator::findLocalMax(vector<double>& values)
{
    vector<int> maxIndexes;

    for (int i = 1; i < values.size() - 1; i++)
    {
        if (values.at(i) > values.at(i - 1) && values.at(i) > values.at(i + 1))
        {
            maxIndexes.push_back(i);
        }
    }
    return maxIndexes;
}


vector<int> FrequencyEstimator::findLocalMin(vector<double>& values)
{
    vector<int> minIndexes;

    for (int i = 1; i < values.size() - 1; i++)
    {
        if (values.at(i) < values.at(i - 1) && values.at(i) < values.at(i + 1))
        {
            minIndexes.push_back(i);
        }
    }
    return minIndexes;
}


bool FrequencyEstimator::frequencyFound(vector<int>& indexes, int windowWidth)
{
    //no frequency can be estimated (too little values)
    if (indexes.empty() || indexes.size() == 1)
    {
        return false;
    }

    //estimate mean distance between local extremes (period)
    int stdDistance = 0;
    for (int i = 0; i < indexes.size() - 1; i++)
    {
        //is not last element
        stdDistance += indexes.at(i + 1) - indexes.at(i);
    }

    stdDistance /= (static_cast<int>(indexes.size()) - 1);

    //check if a common period can be estimated
    for (int i = 0; i < indexes.size() - 1; i++)
    {
        int distance = indexes.at(i + 1) - indexes.at(i);

        if (distance > stdDistance + (windowWidth / 9))
        {
            return false;
        }
    }

    return true;
}


int FrequencyEstimator::getPeriodLenght(vector<int>& indexes)
{
    //estimate mean distance between extremes (period)
    int stdDistance = 0;
    for (int i = 0; i < indexes.size() - 1; i++)
    {
        stdDistance += indexes.at(i + 1) - indexes.at(i);
    }

    stdDistance /= (static_cast<int>(indexes.size()) - 1);
    return stdDistance;
}


cv::Mat FrequencyEstimator::drawFrequencyField(Image* image)
{
    cv::Mat img;
    image->getProcessedImage().copyTo(img);
    int blockSize = image->getBlockSize();
    cv::Mat frequencyField = image->getFrequencyField();

    for (int imgX = 0; imgX < img.cols; imgX++)
    {
        for (int imgY = 0; imgY < img.rows; imgY++)
        {
            int blockCoordX = imgX / blockSize;
            int blockCoordY = imgY / blockSize;
            int correspondingBlockFreq = static_cast<int>(frequencyField.at<double>(blockCoordY, blockCoordX) * 255);
            if (correspondingBlockFreq == -255)
            {
                //frequency could not be estimated for this block
                correspondingBlockFreq = 255; //white
            }
            img.at<unsigned char>(imgY, imgX) = correspondingBlockFreq;
        }
    }

    return img;
}

cv::Mat FrequencyEstimator::extendMatrixSizeBlockwiseAndInterpBorders(const cv::Mat& mat, const cv::Size& size, int blockSize)
{
	cv::Mat freqFieldFullSize = cv::Mat::zeros(size, CV_64F);

	//extend matrix to the size of original image for better averaging
	for (int i = 0; i < freqFieldFullSize.cols; i++) {
		for (int j = 0; j < freqFieldFullSize.rows; j++) {
			double blockFreq = mat.at<double>(j / blockSize, i / blockSize);

			if (blockFreq == -1) {
				for (int u = -1; u <= 1; u++) {
					for (int v = -1; v <= 1; v++) {
						if (Image::isElementInMatSizeRange(i / blockSize + u, j / blockSize + v, mat)) {
							double neighFreqValue = mat.at<double>(j / blockSize + v, i / blockSize + u);
							if (neighFreqValue > blockFreq) {
								blockFreq = neighFreqValue;
							}
						}
					}
				}
			}
			freqFieldFullSize.at<double>(j, i) = blockFreq;
		}
	}

	return freqFieldFullSize;
}


bool FrequencyEstimator::isNotOutOfRange(int index, size_t size)
{
    if (index < 0 || index >= size)
    {
        return false;
    }
    return true;
}


void FrequencyEstimator::smoothenFrequencyField(Image* image)
{
    cv::Mat img = image->getProcessedImage();
    cv::Mat backgroundMask = image->getBackgroundMask();
	cv::Mat frequencyField = image->getFrequencyField();
    int blockSize = image->getBlockSize();

    cv::Mat smoothedFrequencyField = cv::Mat::zeros(img.size(), CV_64F);
    cv::Mat smoothedBlockFrequencyField = cv::Mat::zeros(frequencyField.size(), CV_64F);

    int kernelSize = 2 * blockSize;
    cv::Mat gaussKernel = Filter::get2DGaussianKernel(kernelSize, kernelSize, 20, 20);

	cv::Mat freqFieldFullSize = extendMatrixSizeBlockwiseAndInterpBorders(frequencyField, img.size(), blockSize);

	cv::filter2D(freqFieldFullSize, smoothedFrequencyField, CV_64F, gaussKernel);

    //average values of frequencies per each block
    for (int i = 0; i < frequencyField.cols; i++)
    {
        for (int j = 0; j < frequencyField.rows; j++)
        {
            if (!BackgroundSubstractor::isBackgroundBlock(i, j, backgroundMask))
            {
                smoothedBlockFrequencyField.at<double>(j, i) = OrientationsEstimator::calcAvgForBlock(
                    smoothedFrequencyField, blockSize, i, j);
            }
        }
    }

    image->setFrequencyField(smoothedBlockFrequencyField);
}


vector<double> FrequencyEstimator::smoothenSignatures(vector<double>& xSignatures, int kernelSize, int sigma) {
	cv::Mat gaussKernel = cv::getGaussianKernel(kernelSize, sigma);
	vector<double> newSignatures;

	//for each value at i
	for (int i = 0; i < xSignatures.size(); i++) {
		double newSign = 0;
		for (int j = -kernelSize / 2; j < kernelSize / 2; j++) {
			if (isNotOutOfRange(i + j, xSignatures.size())) {
				//convolve with gaussian kernel
				newSign += xSignatures[i + j] * gaussKernel.at<double>(j + kernelSize / 2);
			}
		}
		newSignatures.push_back(newSign);
	}
	return newSignatures;
}
