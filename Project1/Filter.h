#pragma once

#include <cstddef>
#include "Image.h"

class Filter
{
protected:
    Image srcImage;
    cv::Mat processedImage;

public:
    Filter();
    void setup(Image* image);
    cv::Mat getResultImage();
    virtual void filter();
    static cv::Mat get2DGaussianKernel(int rows, int cols, double sigmax, double sigmay);
};
