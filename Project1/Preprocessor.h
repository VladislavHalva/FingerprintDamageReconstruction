#pragma once

#include "Image.h"

class Preprocessor
{
public:
    Preprocessor();
    void normalize(Image* image);
    void equalize(Image* image); //convert to full range <0, 255>
    void smoothenImage(Image* image, int sigma);
    //cv::Mat calculateMinIntensity(const cv::Mat& mat, int blockSize);
    //cv::Mat calculateMaxIntensity(const cv::Mat& processedImage, int blockSize);
	
    
    static cv::Mat binarize(Image* image);
};
