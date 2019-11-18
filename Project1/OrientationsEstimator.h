#pragma once

#include "Image.h"
#include "ToFieldWrapper.h"
#include "TareaOFieldMapper.h"
#include <vector>

class OrientationsEstimator
{
private:
    cv::Mat thetaX;
    cv::Mat thetaY;

public:
    OrientationsEstimator();
    void estimateFirstOrientationField(Image* image);
    void computeOrientationField(Image* image);
    cv::Mat computeOrientationField(Image* image, int blockSize, cv::Mat* thetaXOut, cv::Mat* thetaYOut);
    void smoothenOrientationField(const cv::Mat& thetaX, const cv::Mat& thetaY, Image* image);
    cv::Mat smoothenOrientationField(const cv::Mat& thetaX, const cv::Mat& thetaY, Image* image, int blockSize,
                                     cv::Mat orientationField);
    
	void smoothenFingerPrintBordersOrientations(cv::Mat* orientationField, cv::Mat* thetaX, cv::Mat* thetaY, const cv::Mat& backgroundMask);
    cv::Point findNeighboringInnerBlock(const vector<cv::Point_<int>>& surroundingBlocks, const cv::Mat& backgroundMask);
   
	double calculateAvgAngleForBlock(int blockCoordX, int blockCoordY, int blockSize, const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& img);

    cv::Mat getThetaX();
    cv::Mat getThetaY();
   
    static cv::Mat calcGradX(cv::Mat& img, int valuesType);
	static cv::Mat calcGradY(cv::Mat& img, int valuesType);

    static double calcAvgForBlock(const cv::Mat& mat, int blockSize, int blockX, int blockY);
	static cv::Mat extendMatrixSizeBlockwise(cv::Mat mat, int blockSize, const cv::Size& size);
    static cv::Mat drawOrientationField(Image* image, bool smooth);
    cv::Mat drawOrientationFieldCustom(Image* image, cv::Mat orientationField, int blockSize);

    void updateOrientationsBasedOnDamage(Image* image);
    void setMostAppropriateOrientationToAreas(Image* image, vector<ToFieldWrapper> customOrientationFields, vector<TareaOFieldMapper> areasOFieldsMapper);
    int getOFieldIndex(int areaIndex, const vector<TareaOFieldMapper>& areasOFieldMapper);
    double getCustomOrientationValueForBlock(const cv::Point& blockPosition, const cv::Mat& customOrientationField,int customBlockSize, int originalBlockSize,
		const cv::Size& imageSize, cv::Mat* thetaX, cv::Mat* thetaY, double* blockThetaXOut, double* blockThetaYOut);
    
	int getValueFromPosition(int x, int y, const int width);
    cv::Point getPositionFromValue(int value, int width);
    int getValueIndexInArray(int value, vector<array<int,2>> arr);
    double getMostFrequentValue(const vector<double>& vec);
};
