#pragma once

#include "Image.h"
#include "BackgroundSubstractor.h"
#include "OrientationDiscontinuityDetector.h"
#include "OCLEstimator.h"
#include "Preprocessor.h"
#include "ImageArea.h"


class DamageDetector {
private:
	Image* image;
	cv::Mat pixelDamageMask;
public:
	DamageDetector();
	bool setup(Image* image);
    bool lowDamageBlockWasNotChangedYet(cv::Mat former, cv::Mat updated, int x, int y);
	void detectDamagedAreas();
    cv::Mat getRidgeQualityMap(cv::Mat odMap, cv::Mat oclMap, cv::Mat ridgeClarityMap, cv::Mat clarityMap, cv::Mat backgroundMask);

	double getOrientationDiscontinuityScore(double value);
	double getOCLScore(double value);
	double getClarityScore(double value);
	double getRidgeClarityScore(double value);
	double estimateOverallQualityFromFeatures(double odScore, double oclScore, double clarityScore, double ridgeClarityScore);

	void setPixelDamageMask(cv::Mat mask);
	cv::Mat getPixelDamageMask();
    static cv::Mat drawQualityMap(Image* img);
}; 

