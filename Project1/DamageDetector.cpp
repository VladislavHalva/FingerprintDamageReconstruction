#include "DamageDetector.h"
#include "RidgeClarityEstimator.h"
#include "ClarityEstimator.h"
#include "FloodFill.h"
#include "ImageArea.h"
#include "HighDamageDetector.h"


DamageDetector::DamageDetector()
{
}


bool DamageDetector::setup(Image* image)
{
    if (image->getProcessedImage().empty()) return false;
    if (image->getNonSmoothedOrientationField().empty()) return false;
    if (image->getBackgroundMask().empty()) return false;

    //copy given image with all features to attributes
    this->image = image;

    return true;
}


void DamageDetector::detectDamagedAreas()
{
	auto orientationsDiscontinuityDetector = new OrientationDiscontinuityDetector();
	cv::Mat odMap = orientationsDiscontinuityDetector->detectDiscontinuities(this->image);

	auto oclEstimator = new OCLEstimator();
	cv::Mat oclMap = oclEstimator->computeOcl(this->image);

	auto ridgeClarityEstimator = new RidgeClarityEstimator();
	cv::Mat ridgeClarityMap = ridgeClarityEstimator->computeRidgeClarity(this->image);

	auto clarityEstimator = new ClarityEstimator();
	cv::Mat clarityMap = clarityEstimator->computeClarity(this->image);

	cv::Mat qualityMap = getRidgeQualityMap(odMap, oclMap, ridgeClarityMap, clarityMap, this->image->getBackgroundMask());
	cv::Mat qualityMapShow;

	this->image->setQualityMap(qualityMap);

	auto highDamageDetector = new HighDamageDetector();
	highDamageDetector->findHeavilyDamagedAreas(this->image);
}


cv::Mat DamageDetector::getRidgeQualityMap(cv::Mat odMap, cv::Mat oclMap, cv::Mat ridgeClarityMap, cv::Mat clarityMap,
    cv::Mat backgroundMask)
{
	cv::Mat qualityMap = cv::Mat::zeros(backgroundMask.size(), CV_64F);

    for(int blockX = 0; blockX < backgroundMask.cols; blockX++)
    {
        for(int blockY = 0; blockY < backgroundMask.rows; blockY++)
        {
            if(BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask))
            {
				qualityMap.at<double>(blockY, blockX) = BACKGROUND;
                continue;
            }

			double odScore = getOrientationDiscontinuityScore(static_cast<double>(odMap.at<unsigned char>(blockY, blockX)));
			double oclScore = getOCLScore(static_cast<double>(oclMap.at<unsigned char>(blockY, blockX)));
			double clarityScore = getClarityScore(static_cast<double>(clarityMap.at<unsigned char>(blockY, blockX)));
			double ridgeClarityScore = getRidgeClarityScore(static_cast<double>(ridgeClarityMap.at<unsigned char>(blockY, blockX)));

			double qualityEstimation = estimateOverallQualityFromFeatures(odScore, oclScore, clarityScore, ridgeClarityScore);
			qualityMap.at<double>(blockY, blockX) = qualityEstimation;
        }
    }

	return qualityMap;
}


double DamageDetector::getOrientationDiscontinuityScore(double value)
{
	if (value == DAMAGED)       return 0;
	if (value == LOW_DAMAGE)    return 0.5;
	if (value == NO_DAMAGE)     return 1;
	else return 0;
}


double DamageDetector::getOCLScore(double value)
{
	if (value > 10) value += 100;
	if (value > 255) value = 255;

	return (255 - value) / 255.; //255 is lowest ocl --> worst image	
}


double DamageDetector::getClarityScore(double value)
{
	if (value == LOW_CLARITY)    return 0;
	if (value == HIGH_CLARITY)  return 1;
	else return 0;
}


double DamageDetector::getRidgeClarityScore(double value)
{
	if (value == LOW_CLARITY)    return 0;
	if (value == HIGH_CLARITY)  return 1;
	else return 0;
}


double DamageDetector::estimateOverallQualityFromFeatures(double odScore, double oclScore, double clarityScore, 
	double ridgeClarityScore)
{
	return 
        4 / 20. * odScore + 
		8 / 20. * oclScore + 
		4 / 20. * ridgeClarityScore + 
		4 / 20. * clarityScore;
}


cv::Mat DamageDetector::drawQualityMap(Image* img) {
	cv::Mat qualityMap = img->getQualityMap();
	cv::Mat qualityMapShow;
	qualityMap.convertTo(qualityMapShow, CV_8U, 255);
	qualityMapShow = Image::extendBlocksToFullSizeImage(qualityMapShow, img->getBlockSize(), img->getProcessedImage().size());
	return qualityMapShow;
}


void DamageDetector::setPixelDamageMask(cv::Mat mask) {
	this->pixelDamageMask = mask;
}


cv::Mat DamageDetector::getPixelDamageMask() {
	return this->pixelDamageMask;
}



