#include "Image.h"
#include "Filter.h"

using namespace std;


Image::Image()
{
    cv::Mat srcImg;
    processedImage = srcImg;
    blockSize = 0;
    windowWidth = 0;
}


Image::Image(cv::Mat srcImage)
{
    //determine size of blocksize and windowWidth for field and frequency map
    blockSize = srcImage.cols / 34;
    (blockSize % 2 == 0) ? blockSize++ : blockSize;
    windowWidth = 2 * blockSize;

    //crop processedImage to multiple of blocksize of field field
    cv::Rect rectCrop(
        0,
        0,
        srcImage.cols / blockSize * blockSize,
        srcImage.rows / blockSize * blockSize);
    cv::Mat croppedImage(srcImage, rectCrop);
	croppedImage.convertTo(this->image, CV_8U);
    croppedImage.convertTo(this->processedImage, CV_8U);
}


bool Image::setSrcImage(cv::Mat image)
{
    try
    {
        //determine size of blocksize
        this->blockSize = image.cols / 19;
        (this->blockSize % 2 == 0) ? this->blockSize++ : this->blockSize;
        this->windowWidth = 2 * this->blockSize;

        cv::Rect rectCrop(
            0,
            0,
            image.cols / this->blockSize * this->blockSize,
            image.rows / this->blockSize * this->blockSize);
        cv::Mat croppedImage(image, rectCrop);
        croppedImage.copyTo(this->image);
		croppedImage.copyTo(this->processedImage);
    }
    catch (...)
    {
        return false;
    }
    return true;
}


bool Image::setProcessedImage(cv::Mat image)
{
    try
    {
        image.copyTo(this->processedImage);
    }
    catch (...)
    {
        return false;
    }
    return true;
}


void Image::setBlockSize(int blockSize)
{
    this->blockSize = blockSize;
}

void Image::setBackgroundMask(cv::Mat bMask)
{
    this->blockBackgroundMask = bMask;
}

void Image::setQualityMap(cv::Mat qMap)
{
	this->qualityMap = qMap;
}

void Image::setHighlyDamagedAreas(vector<ImageArea> areas)
{
	this->higlyDamagedAreas = areas;
}

void Image::setSingularityMap(cv::Mat map)
{
	this->singularityMap = map;
}

void Image::setHighlyDamagedAreasPreview(cv::Mat map)
{
	this->highlyDamagedAreasPreview = map;
}

void Image::setFrequencyField(cv::Mat fField)
{
    this->frequencyField = fField;
}

void Image::setOrientationField(cv::Mat oField)
{
    this->orientationField = oField;
}

void Image::setNonSmoothedOrientationField(cv::Mat oField) {
	this->nonSmoothedOrientationField = oField;
}

void Image::setWindowWidth(int windowWidth)
{
    this->windowWidth = windowWidth;
}


cv::Mat Image::getProcessedImage()
{
    return this->processedImage;
}

cv::Mat Image::getImage()
{
	return this->image;
}

cv::Size Image::getSize()
{
    return this->processedImage.size();
}

cv::Mat Image::getBackgroundMask()
{
    return this->blockBackgroundMask;
}

cv::Mat Image::getQualityMap()
{
	return this->qualityMap;
}

vector<ImageArea> Image::getHighlyDamagedAreas()
{
	return this->higlyDamagedAreas;
}

cv::Mat Image::getSingularityMap()
{
	return this->singularityMap;
}

cv::Mat Image::getHighlyDamagedAreasPreview()
{
	return this->highlyDamagedAreasPreview;
}

int Image::getBlockSize()
{
    return this->blockSize;
}

cv::Mat Image::getFrequencyField()
{
    return this->frequencyField;
}

cv::Mat Image::getOrientationField()
{
    return this->orientationField;
}

cv::Mat Image::getNonSmoothedOrientationField() {
	return this->nonSmoothedOrientationField;
}

int Image::getWindowWidth()
{
    return this->windowWidth;
}


bool Image::isElementInMatSizeRange(int pixelX, int pixelY, const cv::Mat& img)
{
    if (pixelX >= 0 && pixelX < img.cols && pixelY >= 0 && pixelY < img.rows)
    {
        return true;
    }
    return false;
}


bool Image::isElementBorderElementOfMat(int pixelX, int pixelY, const cv::Mat& img)
{
    if (pixelX == 0 || pixelY == 0 || pixelX == (img.cols - 1) || pixelY == (img.rows - 1))
    {
        return true;
    }
    return false;
}


void Image::getPixelBlock(int pixelX, int pixelY, int blockSize, int* pixelBlockX, int* pixelBlockY) {

	*pixelBlockX = pixelX / blockSize;
	*pixelBlockY = pixelY / blockSize;
}


cv::Mat Image::extendBlocksToFullSizeImage(cv::Mat blocks, int blockSize, cv::Size imageSize)
{
	cv::Mat finalImage(imageSize, blocks.type());

    for(int i = 0; i < finalImage.cols; i++)
    {
		for (int j = 0; j < finalImage.rows; j++)
		{
			switch(blocks.type())
			{
			case CV_8U:
				finalImage.at<unsigned char>(j, i) = blocks.at<unsigned char>(j / blockSize, i / blockSize);
				break;
			case CV_64F:
				finalImage.at<double>(j, i) = blocks.at<double>(j / blockSize, i / blockSize);
				break;
			case CV_32S:
				finalImage.at<int>(j, i) = blocks.at<int>(j / blockSize, i / blockSize);
				break;
			}
		}
    }
	return finalImage;
}

cv::Mat_<unsigned char> Image::convertToUCharAndExtendToRange0_255(cv::Mat values)
{
	double min, max;
	cv::minMaxLoc(values, &min, &max);

	cv::Mat convertedValues;
	values.convertTo(convertedValues, CV_8U, 255.0 / (max - min), min * 255.0 / (min - max));
	return convertedValues;
}
