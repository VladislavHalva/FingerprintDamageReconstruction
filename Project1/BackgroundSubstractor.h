#pragma once

#include "Image.h"

#define NOT_VALID 0
#define VALID 1

class BackgroundSubstractor
{
private:
    int backgroundTreshold = 30;

public:
    BackgroundSubstractor();
    void correctInnerBlocksEstimatedAsBackground(Image* image);
    void estimateBackgroundAreaFromVariance(Image* image);
    void interpolateBackgroundMask(Image* image, int cycles);
	cv::Mat colorBackgroundAreasWhite(Image* image);

    static bool isBackgroundBlock(int blockX, int blockY, cv::Mat backgroundMask);
    static bool isBackgroundPixel(int pixelX, int pixelY, Image* image);
	static bool hasBackgroundNeighbor(int blockX, int blockY, const cv::Mat backgroundMask, const cv::Mat validityMask);
    static bool hasBackgroundNeighbor(int blockX, int blockY, const cv::Mat& backgroundMask);
    static cv::Mat drawBackground(Image* image);
};
