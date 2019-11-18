#include "OrientationsEstimator.h"
#include "Filter.h"
#include "ToFieldWrapper.h"
#include "TareaOFieldMapper.h"
#include "BackgroundSubstractor.h"
#include "SingularityDetector.h"
#include "BasicOperations.h"


OrientationsEstimator::OrientationsEstimator()
{
}


void OrientationsEstimator::computeOrientationField(Image* image)
{
    cv::Mat img = image->getProcessedImage();
    cv::Size imageSize = img.size();
    int blockSize = image->getBlockSize();
    cv::Mat orientationField(img.rows / blockSize, img.cols / blockSize, CV_64F);

    //continuous fields for smoothing
    cv::Mat thetaX(img.rows / blockSize, img.cols / blockSize, CV_64F);
    cv::Mat thetaY(img.rows / blockSize, img.cols / blockSize, CV_64F);

    //compute gradients
    cv::Mat gradX = calcGradX(img, CV_64F);
    cv::Mat gradY = calcGradY(img, CV_64F);

    //calc field for each block at (i,j)
    for (int blockX = 0; blockX < orientationField.cols; blockX++)
    {
        for (int blockY = 0; blockY < orientationField.rows; blockY++)
        {
            double angle = calculateAvgAngleForBlock(blockX, blockY, blockSize, gradX, gradY, img);
            orientationField.at<double>(blockY, blockX) = angle;
            //vector elements used for smoothing
            thetaX.at<double>(blockY, blockX) = cos(2. * (angle * CV_PI / 180.));
            thetaY.at<double>(blockY, blockX) = sin(2. * (angle * CV_PI / 180.));
        }
    }

	smoothenFingerPrintBordersOrientations(&orientationField, &thetaX, &thetaY, image->getBackgroundMask());

    thetaX.copyTo(this->thetaX);
    thetaY.copyTo(this->thetaY);
	thetaX.copyTo(image->thetaX);
	thetaY.copyTo(image->thetaY);

    image->setNonSmoothedOrientationField(orientationField);
	image->setOrientationField(orientationField);
}


cv::Mat OrientationsEstimator::computeOrientationField(Image* image, int blockSize, cv::Mat* thetaXOut, cv::Mat* thetaYOut)
{
	cv::Mat img = image->getProcessedImage();
	cv::Size imageSize = img.size();
	int fieldsSize = (img.rows / blockSize == 0) ? img.rows / blockSize : (img.rows / blockSize) + 1;
	cv::Mat orientationField(fieldsSize, fieldsSize, CV_64F);

	//continuous fields for smoothing
	cv::Mat thetaX(fieldsSize, fieldsSize, CV_64F);
	cv::Mat thetaY(fieldsSize, fieldsSize, CV_64F);

	//compute gradients
	cv::Mat gradX = calcGradX(img, CV_64F);
	cv::Mat gradY = calcGradY(img, CV_64F);

	//calc field for each block at (i,j)
	for (int i = 0; i < orientationField.cols; i++)
	{
		for (int j = 0; j < orientationField.rows; j++)
		{
			double angle = calculateAvgAngleForBlock(i, j, blockSize, gradX, gradY, img);
			orientationField.at<double>(j, i) = angle;
			//vector elements used for smoothing
			thetaX.at<double>(j, i) = cos(2. * (angle * CV_PI / 180.));
			thetaY.at<double>(j, i) = sin(2. * (angle * CV_PI / 180.));
		}
	}

	thetaX.copyTo(*thetaXOut);
	thetaY.copyTo(*thetaYOut);
	return orientationField;
}


cv::Mat OrientationsEstimator::drawOrientationField(Image* image, bool smooth)
{
	cv::Mat orientationField;
	int blockSize = image->getBlockSize();

    if(smooth)
    {
		orientationField = image->getOrientationField();
    }
	else
	{
		orientationField = image->getNonSmoothedOrientationField();
	}

    cv::Mat finalImage;
    image->getProcessedImage().copyTo(finalImage);

    //darker 
	finalImage /= 2;

    for (int oFieldCoordX = 0; oFieldCoordX < orientationField.cols; oFieldCoordX++)
    {
        for (int oFieldCoordY = 0; oFieldCoordY < orientationField.rows; oFieldCoordY++)
        {
            cv::Point lineStartPoint(
                static_cast<int>(oFieldCoordX * blockSize),
                static_cast<int>(oFieldCoordY * blockSize));

            cv::Scalar vector(
                //needs to be converted from degrees to radians
                cos(orientationField.at<double>(oFieldCoordY, oFieldCoordX) * CV_PI / 180.0),
                sin(orientationField.at<double>(oFieldCoordY, oFieldCoordX) * CV_PI / 180.0));

            cv::Point lineEndPoint(
                static_cast<int>(lineStartPoint.x + (blockSize * vector.val[0])),
                static_cast<int>(lineStartPoint.y + (blockSize * vector.val[1]))
            );

            line(finalImage, lineStartPoint, lineEndPoint, cv::Scalar(255));
        }
    }

    return finalImage;
}


cv::Mat OrientationsEstimator::drawOrientationFieldCustom(Image* image, cv::Mat orientationField, int blockSize)
{
	cv::Mat finalImage;
	image->getProcessedImage().copyTo(finalImage);

	//darker 
	finalImage /= 2;

	for (int oFieldCoordX = 0; oFieldCoordX < orientationField.cols; oFieldCoordX++)
	{
		for (int oFieldCoordY = 0; oFieldCoordY < orientationField.rows; oFieldCoordY++)
		{
			cv::Point lineStartPoint(
				static_cast<int>(oFieldCoordX * blockSize),
				static_cast<int>(oFieldCoordY * blockSize));

			cv::Scalar vector(
				//needs to be converted from degrees to radians
				cos(orientationField.at<double>(oFieldCoordY, oFieldCoordX) * CV_PI / 180.0),
				sin(orientationField.at<double>(oFieldCoordY, oFieldCoordX) * CV_PI / 180.0));

			cv::Point lineEndPoint(
				static_cast<int>(lineStartPoint.x + (blockSize * vector.val[0])),
				static_cast<int>(lineStartPoint.y + (blockSize * vector.val[1]))
			);

			line(finalImage, lineStartPoint, lineEndPoint, cv::Scalar(255));
		}
	}

	return finalImage;
}



cv::Mat OrientationsEstimator::extendMatrixSizeBlockwise(cv::Mat mat, int blockSize, const cv::Size& size)
{
	cv::Mat matExtendedSize = cv::Mat::zeros(size, CV_64F);

	for (int i = 0; i < matExtendedSize.cols; i++) {
		for (int j = 0; j < matExtendedSize.rows; j++) {
			matExtendedSize.at<double>(j, i) = mat.at<double>(j / blockSize, i / blockSize);
		}
	}

	return matExtendedSize;
}


void OrientationsEstimator::smoothenOrientationField(const cv::Mat& thetaX, const cv::Mat& thetaY,
                                                     Image* image)
{
    cv::Mat orientationField = image->getNonSmoothedOrientationField();
    cv::Mat img = image->getProcessedImage();
    int blockSize = image->getBlockSize();

    //get 2D gaussian kernel
    int kernelSize = 5 * blockSize;
    cv::Mat gaussKernel = Filter::get2DGaussianKernel(kernelSize, kernelSize, 20, 20);

    cv::Mat smoothedThetaX = cv::Mat::zeros(img.size(), CV_64F);
    cv::Mat smoothedThetaY = cv::Mat::zeros(img.size(), CV_64F);
    cv::Mat smoothedOrientationField;
    orientationField.copyTo(smoothedOrientationField);

	//extend matrix to the size of original image for better averaging 
	cv::Mat thetaXFullSize = extendMatrixSizeBlockwise(thetaX, blockSize, img.size());
    cv::Mat thetaYFullSize = extendMatrixSizeBlockwise(thetaY, blockSize, img.size());

    //convolve
    filter2D(thetaXFullSize, smoothedThetaX, CV_64F, gaussKernel);
    filter2D(thetaYFullSize, smoothedThetaY, CV_64F, gaussKernel);

    //average values of field vectors for each block non-weighted averaging
    for (int i = 0; i < orientationField.cols; i++)
    {
        for (int j = 0; j < orientationField.rows; j++)
        {
            double smoothedThetaXBlock = calcAvgForBlock(smoothedThetaX, blockSize, i, j);
            double smoothedThetaYBlock = calcAvgForBlock(smoothedThetaY, blockSize, i, j);

			//save in image 
			image->thetaX.at<double>(j, i) = smoothedThetaXBlock;
			image->thetaY.at<double>(j, i) = smoothedThetaYBlock;

            //convert field vector to angle
            smoothedOrientationField.at<double>(j, i) = 0.5 * cv::fastAtan2(smoothedThetaYBlock, smoothedThetaXBlock);
        }
    }
    image->setOrientationField(smoothedOrientationField);
}


cv::Mat OrientationsEstimator::smoothenOrientationField(const cv::Mat& thetaX, const cv::Mat& thetaY,
	Image* image, int blockSize, cv::Mat orientationField)
{
	cv::Mat img = image->getProcessedImage();

	//get 2D gaussian kernel
	int kernelSize = 2 * blockSize;
	cv::Mat gaussKernel = Filter::get2DGaussianKernel(kernelSize, kernelSize, 20, 20);

	cv::Mat smoothedThetaX = cv::Mat::zeros(img.size(), CV_64F);
	cv::Mat smoothedThetaY = cv::Mat::zeros(img.size(), CV_64F);
	cv::Mat smoothedOrientationField;
	orientationField.copyTo(smoothedOrientationField);

	//extend matrix to the size of original image for better averaging 
	cv::Mat thetaXFullSize = extendMatrixSizeBlockwise(thetaX, blockSize, img.size());
	cv::Mat thetaYFullSize = extendMatrixSizeBlockwise(thetaY, blockSize, img.size());

	//convolve
	filter2D(thetaXFullSize, smoothedThetaX, CV_64F, gaussKernel);
	filter2D(thetaYFullSize, smoothedThetaY, CV_64F, gaussKernel);

	//average values of field vectors for each block non-weighted averaging
	for (int i = 0; i < orientationField.cols; i++)
	{
		for (int j = 0; j < orientationField.rows; j++)
		{
			double smoothedThetaXBlock = calcAvgForBlock(smoothedThetaX, blockSize, i, j);
			double smoothedThetaYBlock = calcAvgForBlock(smoothedThetaY, blockSize, i, j);
			
			//convert field vector to angle
			smoothedOrientationField.at<double>(j, i) = 0.5 * cv::fastAtan2(smoothedThetaYBlock, smoothedThetaXBlock);
		}
	}

	return smoothedOrientationField;
}


void OrientationsEstimator::smoothenFingerPrintBordersOrientations(cv::Mat* orientationField, cv::Mat* thetaX,
	cv::Mat* thetaY, const cv::Mat& backgroundMask)
{
	cv::Mat show = cv::Mat::zeros(orientationField->size(), CV_8U);

	for(int blockX = 0; blockX < orientationField->cols; blockX++)
	{
		for(int blockY = 0; blockY < orientationField->rows; blockY++)
		{
			if(BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask))
			{
				continue;
			}
			
			if (BackgroundSubstractor::hasBackgroundNeighbor(blockX, blockY, backgroundMask))
			{
				show.at<unsigned char>(blockY, blockX) = 127;

				vector<cv::Point> area3x3 = BasicOperations::getSurroundingPoints(blockX, blockY);

				//find neighboring inner block, its value will be copied to border block and surrounding background
				cv::Point innerBlock = findNeighboringInnerBlock(area3x3, backgroundMask);
							
				//no value to be copied, does not have inner block as neighbor
				if (innerBlock.x == 0 && innerBlock.y == 0) continue;

				show.at<unsigned char>(innerBlock) = 200;

				//include self
				area3x3.push_back(cv::Point(blockX, blockY));

				for (auto block : area3x3)
				{
					if (Image::isElementInMatSizeRange(block.x, block.y, backgroundMask))
					{
						if (BackgroundSubstractor::isBackgroundBlock(block.x, block.y, backgroundMask) ||
							BackgroundSubstractor::hasBackgroundNeighbor(block.x, block.y, backgroundMask))
						{
							orientationField->at<double>(block) = orientationField->at<double>(innerBlock);
							thetaX->at<double>(block) = thetaX->at<double>(innerBlock);
							thetaY->at<double>(block) = thetaY->at<double>(innerBlock);
						}
					}
				}
			}
		}
	}
}

cv::Point OrientationsEstimator::findNeighboringInnerBlock(const vector<cv::Point_<int>>& surroundingBlocks,
	const cv::Mat& backgroundMask)
{
	auto innerBlock = cv::Point(0, 0);
	//find neighbor that is not border or background
	for (auto block : surroundingBlocks)
	{
		if (Image::isElementInMatSizeRange(block.x, block.y, backgroundMask))
		{
			if (!BackgroundSubstractor::isBackgroundBlock(block.x, block.y, backgroundMask) &&
				!BackgroundSubstractor::hasBackgroundNeighbor(block.x, block.y, backgroundMask))
			{
				innerBlock = block;
				break;
			}
		}
	}

	return innerBlock;
}


cv::Mat OrientationsEstimator::calcGradX(cv::Mat& img, int valuesType)
{
    cv::Mat gradX(img.size(), valuesType);
    Scharr(img, gradX, valuesType, 1, 0, 5);
    return gradX;
}


cv::Mat OrientationsEstimator::calcGradY(cv::Mat& img, int valuesType)
{
    cv::Mat gradY(img.size(), valuesType);
    Scharr(img, gradY, valuesType, 0, 1, 5);
    return gradY;
}


double OrientationsEstimator::calculateAvgAngleForBlock(int blockCoordX, int blockCoordY, int blockSize,
                                                        const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& img)
{
    double Vx = 0;
    double Vy = 0;

    //iterate over pixel belonging to current block + calculate Vx and Vy features
	int limitX = (blockCoordX * blockSize + blockSize <= img.cols) ? blockCoordX * blockSize + blockSize : img.cols;
	int limitY = (blockCoordY * blockSize + blockSize <= img.rows) ? blockCoordY * blockSize + blockSize : img.rows;

    for (int u = blockCoordX * blockSize; u < limitX; u++)
    {
        for (int v = blockCoordY * blockSize; v < limitY; v++)
        {
            double pixelGradX = gradX.at<double>(v, u);
            double pixelGradY = gradY.at<double>(v, u);
            Vx += (2 * pixelGradX * pixelGradY);
            Vy += (pixelGradX * pixelGradX - pixelGradY * pixelGradY);
        }
    }

    //estimate angle from calculated features
    double angle = 0.;

    if (Vx != 0)
    {
        angle = 90.0 + (0.5 * cv::fastAtan2(Vx, Vy));
    }

    return angle;
}


double OrientationsEstimator::calcAvgForBlock(const cv::Mat& mat, int blockSize, int blockX, int blockY)
{
    //calculate avg only for one block specified by X and Y coordination (in blockField)
    double sum = 0;

	int limitX = (blockX * blockSize + blockSize < mat.cols) ? blockX * blockSize + blockSize : mat.cols - 1;
	int limitY = (blockY * blockSize + blockSize < mat.rows) ? blockY * blockSize + blockSize : mat.rows - 1;

    for (int i = blockX * blockSize; i < limitX; i++)
    {
        for (int j = blockY * blockSize; j < limitY; j++)
        {
            sum += mat.at<double>(j, i);
        }
    }
    sum /= (blockSize * blockSize);
    return sum;
}


cv::Mat OrientationsEstimator::getThetaX()
{
    return this->thetaX;
}


cv::Mat OrientationsEstimator::getThetaY()
{
    return this->thetaY;
}


void OrientationsEstimator::updateOrientationsBasedOnDamage(Image* image)
{
	vector<ImageArea> damageAreas = image->getHighlyDamagedAreas();
	vector<int> damageSizes;

	//no damaged areas
	if(damageAreas.empty())
		return;

	//get areas sizes
	for (ImageArea damageArea : damageAreas)
	{
		int smallerDimension = (damageArea.getHeight() < damageArea.getWidth()) ? damageArea.getHeight() : damageArea.getWidth();
		damageSizes.push_back(smallerDimension);
	}

	vector<int>::iterator maxSize;
	vector<int>::iterator minSize;
	maxSize = max_element(damageSizes.begin(), damageSizes.end());
	minSize = min_element(damageSizes.begin(), damageSizes.end());

	//size of range of damage areas sizes, for which one oField will be generated
	int rangeSize = 2;
	auto oEstimator = new OrientationsEstimator();
	vector<ToFieldWrapper> customOrientationFields;
	vector<TAreaOFieldMapper> areasOFieldsMapper;

	//generate oFields for different sizes od damaged areas
	for(int rangeBegin = 1; rangeBegin <= *maxSize; rangeBegin += rangeSize)
	{
		bool areaOfSizeInCurrentRangeFound = false;

		for (int sizeIndex = 0; sizeIndex < damageSizes.size(); sizeIndex++)
		{
			int size = damageSizes.at(sizeIndex);

			if(size >= rangeBegin && size <= rangeBegin + rangeSize - 1)
			{
				//area size is in current range
				cv::Mat orientationField;
				int oFieldIndex;

				if(areaOfSizeInCurrentRangeFound)
				{
					//oField of current size has already been generated
					for (int i = 0; i < customOrientationFields.size(); i++)
					{
						ToFieldWrapper oFieldWrapper = customOrientationFields.at(i);

						if(oFieldWrapper.rangeBegin == rangeBegin)
						{
							orientationField = oFieldWrapper.field;
							oFieldIndex = i;
							break;
						}
					}				
				}
				else
				{
					//create oField
					int customBlockSize = (image->getBlockSize() * (rangeBegin + 1)) / 2;
					customBlockSize = (customBlockSize % 2 == 0) ? customBlockSize + 1 : customBlockSize;
					cv::Mat thetaX;
					cv::Mat thetaY;
					
					orientationField = oEstimator->computeOrientationField(image, customBlockSize, &thetaX, &thetaY);
					orientationField = oEstimator->smoothenOrientationField(thetaX, thetaY, image, customBlockSize, orientationField);

					areaOfSizeInCurrentRangeFound = true;

					//add oField to bank
					ToFieldWrapper oFieldWrapper{
						rangeBegin,
						rangeBegin + rangeSize - 1,
						customBlockSize,
						orientationField,
						thetaX,
						thetaY
					};
					customOrientationFields.push_back(oFieldWrapper);
					oFieldIndex = customOrientationFields.size() - 1;
				}

				//add area - oField combination to mapper
				TAreaOFieldMapper mapper {
					sizeIndex,
					oFieldIndex
				};
				areasOFieldsMapper.push_back(mapper);

				//cv::Mat show = drawOrientationFieldCustom(image, smoothOrientationField, customBlockSize);	
				//cv::imshow("custom", show);
				//cv::waitKey(1000);
			}
		}
	}

	setMostAppropriateOrientationToAreas(image, customOrientationFields, areasOFieldsMapper);
}


void OrientationsEstimator::setMostAppropriateOrientationToAreas(Image* image,
	vector<ToFieldWrapper> customOrientationFields, vector<TareaOFieldMapper> areasOFieldsMapper)
{
	vector<ImageArea> damagedAreas = image->getHighlyDamagedAreas();
	cv::Mat orientationField = image->getOrientationField();

	//update orientations in all damaged areas
	for(int areaIndex = 0; areaIndex < damagedAreas.size(); areaIndex++)
	{
		//skip singularity areas, they are supposed to have high curvature --> seems like low quality
		if (damagedAreas.at(areaIndex).getPointsState() == SINGULARITY)
			continue;
		
		int oFieldIndex = getOFieldIndex(areaIndex, areasOFieldsMapper);
		if (oFieldIndex == -1) return;

		cv::Mat customOrientationField = customOrientationFields.at(oFieldIndex).field;

		//change field for all blocks that belong to current area
		for (cv::Point blockPosition : damagedAreas.at(areaIndex).getPoints())
		{
			double blockThetaX;
			double blockThetaY;
			double newOrientation = getCustomOrientationValueForBlock(blockPosition, customOrientationField, customOrientationFields.at(oFieldIndex).blockSize, 
				image->getBlockSize(), image->getSize(), &thetaX, &thetaY, &blockThetaX, &blockThetaY);

			orientationField.at<double>(blockPosition) = newOrientation;
			image->thetaX.at<double>(blockPosition) = blockThetaX;
			image->thetaY.at<double>(blockPosition) = blockThetaY;
		}
	}

	image->setOrientationField(orientationField);
}


double OrientationsEstimator::getCustomOrientationValueForBlock(const cv::Point& blockPosition,
	const cv::Mat& customOrientationField, int customBlockSize, int originalBlockSize, const cv::Size& imageSize, 
	cv::Mat* thetaX, cv::Mat* thetaY, double* blockThetaXOut, double* blockThetaYOut)
{
	if(originalBlockSize == customBlockSize)
	{
		//no change in blockSize
		return customOrientationField.at<double>(blockPosition.y, blockPosition.x);
	}

	vector<double> orientations;
	vector<double> thetasX;
	vector<double> thetasY;

	cv::Mat originalBlockArea = cv::Mat::zeros(cv::Size(originalBlockSize, originalBlockSize), CV_64F);
	for(int blockPixelX = 0; blockPixelX < originalBlockArea.cols; blockPixelX++)
	{
		for (int blockPixelY = 0; blockPixelY < originalBlockArea.rows; blockPixelY++)
		{
			int pixelX = originalBlockSize * blockPosition.x + blockPixelX;
			int pixelY = originalBlockSize * blockPosition.y + blockPixelY;

			int customBlockX = pixelX / customBlockSize;
			int customBlockY = pixelY / customBlockSize;

			orientations.push_back(customOrientationField.at<double>(customBlockY, customBlockX));
			thetasX.push_back(thetaX->at<double>(customBlockY, customBlockX));
			thetasY.push_back(thetaY->at<double>(customBlockY, customBlockX));
		}
	}

	double mFreqThetaX = getMostFrequentValue(thetasX);
	double mFreqThetaY = getMostFrequentValue(thetasY);
	*blockThetaXOut = mFreqThetaX;
	*blockThetaYOut = mFreqThetaY;

	double mFrequent = getMostFrequentValue(orientations);
	return mFrequent;
}

int OrientationsEstimator::getValueFromPosition(int x, int y, const int width)
{
	return x * width + y;
}

cv::Point OrientationsEstimator::getPositionFromValue(int value, int width)
{
	return cv::Point(value / width, value % width);
}

int OrientationsEstimator::getValueIndexInArray(int value, vector<array<int,2>> arr)
{
	for(int i = 0; i < arr.size(); i++)
	{
		if (arr.at(i)[0] == value)
			return i;
	}
	return -1;
}

double OrientationsEstimator::getMostFrequentValue(const vector<double>& vec)
{
	struct valueCount
	{
		int value;
		int count;
	};

	vector<struct valueCount> valueCounts;

	for (double value : vec)
	{
		bool found = false;

		for (struct valueCount vCount : valueCounts)
		{
			if (vCount.value == value)
			{
				vCount.count += 1;
				found = true;
				break;
			}
		}

		if(!found)
		{
			struct valueCount vCount {value, 1};
			valueCounts.push_back(vCount);
		}
	}

	int mFrequentValue = valueCounts.at(0).value;
	int frequency = valueCounts.at(0).count;
	for(int i = 0; i < valueCounts.size(); i++)
	{
		if(valueCounts.at(i).count > frequency)
		{
			frequency = valueCounts.at(i).count;
			mFrequentValue = valueCounts.at(i).value;
		}
	}

	return mFrequentValue;
}


int OrientationsEstimator::getOFieldIndex(int areaIndex, const vector<TareaOFieldMapper>& areasOFieldMapper)
{
	for (TareaOFieldMapper mapper : areasOFieldMapper)
	{
		if (mapper.areaIndex == areaIndex) {
			return mapper.oFieldIndex;
		}
	}
	return -1;
}

