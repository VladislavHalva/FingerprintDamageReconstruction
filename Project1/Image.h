#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>
#include <math.h>
#include <iostream>
#include "ImagePointState.h"
#include "ImageArea.h"

#define DEBUG 1

using namespace std;

class Image
{
private:
	cv::Mat image;
    cv::Mat processedImage;
    
	cv::Mat orientationField;
	cv::Mat nonSmoothedOrientationField;
	
    cv::Mat frequencyField;
    
	cv::Mat blockBackgroundMask;
	
	cv::Mat singularityMap;
	
	cv::Mat qualityMap;
	vector<ImageArea> higlyDamagedAreas;
	cv::Mat highlyDamagedAreasPreview;
    
	int blockSize;
    int windowWidth;

public:
    Image();
    Image(cv::Mat srcImage);

    bool setSrcImage(cv::Mat image);
    bool setProcessedImage(cv::Mat image);
    void setBlockSize(int blockSize);
    void setWindowWidth(int windowWidth);
    void setOrientationField(cv::Mat oField);
	void setNonSmoothedOrientationField(cv::Mat oField);
    void setFrequencyField(cv::Mat fField);
    void setBackgroundMask(cv::Mat bMask);
	void setQualityMap(cv::Mat qMap);
	void setHighlyDamagedAreas(vector<ImageArea> areas);
	void setSingularityMap(cv::Mat map);
	void setHighlyDamagedAreasPreview(cv::Mat map);

    cv::Mat getProcessedImage();
	cv::Mat getImage();
    cv::Size getSize();
    int getBlockSize();
    int getWindowWidth();
    cv::Mat getOrientationField();
	cv::Mat getNonSmoothedOrientationField();
    cv::Mat getFrequencyField();
    cv::Mat getBackgroundMask();
	cv::Mat getQualityMap();
	vector<ImageArea> getHighlyDamagedAreas();
	cv::Mat getSingularityMap();
	cv::Mat getHighlyDamagedAreasPreview();

    static bool isElementInMatSizeRange(int pixelX, int pixelY, const cv::Mat& img);
    static bool isElementBorderElementOfMat(int pixelX, int pixelY, const cv::Mat& img);
    static void getPixelBlock(int pixelX, int pixelY, int blockSize, int* pixelBlockX, int* pixelBlockY);
	static cv::Mat extendBlocksToFullSizeImage(cv::Mat blocks, int blockSize, cv::Size imageSize);
	static cv::Mat_<unsigned char> convertToUCharAndExtendToRange0_255(cv::Mat values);

	cv::Mat thetaX;
	cv::Mat thetaY;
};
