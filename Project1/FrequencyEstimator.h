#pragma once

#include "Image.h"

class FrequencyEstimator
{
private:
    vector<int> findLocalMax(vector<double>& values);
	vector<int> findLocalMin(vector<double>& values);
    bool frequencyFound(vector<int>& indexes, int windowWidth);
    int getPeriodLenght(vector<int>& indexes);
    void interpolateFreqField(cv::Mat img, cv::Mat frequencyField, int blockSize, cv::Mat backgroundMask);
public:
    FrequencyEstimator();
    bool isNotOutOfRange(int index, size_t size);
    vector<double> smoothenSignatures(vector<double>& xSignatures, int kernelSize, int sigma);
    void computeFrequencyField(Image* image);
    cv::Mat extendMatrixSizeBlockwiseAndInterpBorders(const cv::Mat& mat, const cv::Size& size, int blockSize);
    void smoothenFrequencyField(Image* image);
    static cv::Mat drawFrequencyField(Image* image);
};
