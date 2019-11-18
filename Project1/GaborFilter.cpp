#include "GaborFilter.h"
#include "OrientationsEstimator.h"
#include "BackgroundSubstractor.h"


GaborFilter::GaborFilter()
{
}


void GaborFilter::setup(Image* image)
{
	//copy given image with all features to attributes
    this->srcImage = *image;

	createBankOfGaborFilters();
}

void GaborFilter::createBankOfGaborFilters()
{
	cv::Mat orientationField = this->srcImage.getOrientationField();
	cv::Mat frequencyField = this->srcImage.getFrequencyField();
	cv::Mat backgroundMask = this->srcImage.getBackgroundMask();

	double maxOrientation = getMaxInMatWithValidityMask(orientationField, backgroundMask);
	double minOrientation = getMinInMatWithValidityMask(orientationField, backgroundMask);
	double maxFrequency = getMaxInMatWithValidityMask(frequencyField, backgroundMask);
	double minFrequency = getMinInMatWithValidityMask(frequencyField, backgroundMask);

	//determine step between field and frequency of filters in bank
	double bankOrientationStep = (maxOrientation - minOrientation) / BANK_SIZE;
	double bankFrequencyStep = (maxFrequency - minFrequency) / BANK_SIZE;

	//used to save filters parameters 
	vector<double> bankFiltersOrientations;
	vector<double> bankFiltersFrequencies;

	//create filters in bank one after another
	double currentFilterOrient = 0.;

	for (int bankX = 0; bankX < BANK_SIZE; bankX++) {
		double currentFilterFreq = 0.;
		bankFiltersOrientations.push_back(currentFilterOrient);

		for (int bankY = 0; bankY < BANK_SIZE; bankY++) {
			bankFiltersFrequencies.push_back(currentFilterFreq);

			//create gabor kernel according to field and frequency 
			double currentFilterOrientNormalRad = (currentFilterOrient - 90) * CV_PI / 180;
			cv::Mat currentFilter = cv::getGaborKernel(this->kernelSize, this->stdDev,
				currentFilterOrientNormalRad, 1. / currentFilterFreq, this->aspectRatio, this->offset, CV_64F);
			currentFilter.copyTo(this->gaborFilterBank[bankX][bankY]);

			//cv::imshow("a", currentFilter);
			//cv::waitKey(20);

			currentFilterFreq += bankFrequencyStep;
		}
		currentFilterOrient += bankOrientationStep;
	}

	//save filters parameters
	this->bankFiltersOrientations = bankFiltersOrientations;
	this->bankFiltersFrequencies = bankFiltersFrequencies;
}

void GaborFilter::filter() 
{
	cv::Mat sourceImage;
	this->srcImage.getProcessedImage().convertTo(sourceImage, CV_64F);
	cv::Mat processedImage = cv::Mat::zeros(sourceImage.size(), CV_8U);
	cv::Mat orientationField = this->srcImage.getOrientationField();
	cv::Mat frequencyField = this->srcImage.getFrequencyField();
	cv::Mat backgroundMask = this->srcImage.getBackgroundMask();
	cv::Mat qualityMap = this->srcImage.getQualityMap();
	int blockSize = this->srcImage.getBlockSize();

    //filter
    for(int blockX = 0; blockX < sourceImage.cols / blockSize; blockX++)
    {
		for (int blockY = 0; blockY < sourceImage.rows / blockSize; blockY++)
		{
            if(BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask))
            {   
                //skip background
				continue;
            }

			//extract only currently processed block
			auto blockBoundaries = cv::Rect(blockX * blockSize, blockY * blockSize, blockSize, blockSize);
			cv::Mat extractedBlock = sourceImage(blockBoundaries);

            if(qualityMap.at<double>(blockY, blockX) > 0.5)
            {
				double min, max;
				cv::minMaxLoc(extractedBlock, &min, &max);

				extractedBlock.convertTo(processedImage(blockBoundaries), CV_8U, 255.0 / (max - min), min * 255.0 / (min - max));
            }
			else {
				//choose the right filter kernel according to field and frequency
				double blockOrientation = orientationField.at<double>(blockY, blockX);
				double blockFrequency = frequencyField.at<double>(blockY, blockX);

				int closestOrientIndex = getClosestValueIndex(blockOrientation, this->bankFiltersOrientations);
				int closestFreqIndex = getClosestValueIndex(blockFrequency, this->bankFiltersFrequencies);

				cv::Mat gaborKernel = this->gaborFilterBank[closestOrientIndex][closestFreqIndex];

				//filter block
				cv::Mat filteredBlock;
				cv::filter2D(extractedBlock, filteredBlock, CV_64F, gaborKernel);

				//convert to needed range 0 - 255
				filteredBlock = abs(filteredBlock);

				double min, max;
				cv::minMaxLoc(filteredBlock, &min, &max);

				cv::Mat convertedFilteredBlock;
				filteredBlock.convertTo(convertedFilteredBlock, CV_8U, 255.0 / (max - min), min * 255.0 / (min - max));

				//save filtered block to processed image matrix
				convertedFilteredBlock.copyTo(processedImage(blockBoundaries));
			}
		}
    }

	processedImage.copyTo(this->processedImage);
}


int GaborFilter::getClosestValueIndex(double value, const vector<double>& vector)
{
	bool firstValue = true;
	double smallestDiff;
	int closestIndex;

    for(int i = 0; i < vector.size(); i++)
    {
        if(firstValue)
        {
			smallestDiff = abs(vector.at(i) - value);
			closestIndex = i;
			firstValue = false;
        }

        if(abs(vector.at(i) - value) < smallestDiff)
        {
			smallestDiff = abs(vector.at(i) - value);
			closestIndex = i;
        }
    }

	return closestIndex;
}


double GaborFilter::getMaxInMatWithValidityMask(const cv::Mat& mat, const cv::Mat& validityMask) {
	double max = 0.;
	bool firstValid = true;

	for (int i = 0; i < mat.cols; i++) {
		for (int j = 0; j < mat.rows; j++) {
			if ((int)validityMask.at<unsigned char>(j, i) == 0) {
				if (firstValid) {
					max = mat.at<double>(j, i);
					firstValid = false;
				}

				if (mat.at<double>(j, i) > max) {
					max = mat.at<double>(j, i);
				}
			}
		}
	}

	return max;
}


double GaborFilter::getMinInMatWithValidityMask(const cv::Mat& mat, const cv::Mat& validityMask) {
	double min = 0.;
	bool firstValid = true;

	for (int i = 0; i < mat.cols; i++) {
		for (int j = 0; j < mat.rows; j++) {
			if ((int)validityMask.at<unsigned char>(j, i) == 0) {
				if (firstValid) {
					min = mat.at<double>(j, i);
					firstValid = false;
				}

				if (mat.at<double>(j, i) < min) {
					min = mat.at<double>(j, i);
				}
			}
		}
	}

	return min;
}
