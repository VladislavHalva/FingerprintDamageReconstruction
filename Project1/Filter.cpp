#include "Filter.h"


Filter::Filter()
{
}


void Filter::setup(Image* image)
{
    this->srcImage = *image;
}


cv::Mat Filter::getResultImage()
{
    return this->processedImage;
}


void Filter::filter()
{
    //no implementation,  abstract 
}


cv::Mat Filter::get2DGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
    auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_64F);
    auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_64F);
    return gauss_x * gauss_y.t();
}
