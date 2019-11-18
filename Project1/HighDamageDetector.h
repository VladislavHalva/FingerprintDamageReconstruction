#pragma once
#include "Image.h"


#define CHECKED 1
#define OUTER_BLOCK 0
#define INNER_BLOCK 127
#define BORDER_BLOCK 200
#define LOW_DAMAGE_EXT 32000
#define DAMAGED_EXT 32001

class HighDamageDetector
{
public:
	HighDamageDetector();
	void findHeavilyDamagedAreas(Image* image);
	cv::Mat divideQualityMapToCategories(cv::Mat qualityMap, const cv::Mat& backgroundMask);
	cv::Mat findContinuousAreasInQualityMap(cv::Mat qualityCategoryMap, const cv::Mat backgroundMask);
	cv::Mat identifyAreas(cv::Mat areasClasificationMap, cv::Mat backgroundMask, vector<ImageArea>* damagedAreas);
	int convertClassificationValueToExt(cv::Mat* identifiedAreas);
	void removeLowerDamageAreasNotConnectedToHiglyDamagedArea(cv::Mat* identifiedAreas, const cv::Mat& backgroundMask,
		int* numberOfLowerDamageAreas);
	vector<ImageArea> fillHighlyDamagedAreas(cv::Mat* identifiedAreas, const cv::Mat& backgroundMask, int* areaIndex);
	void attachLowerDamageAreas(cv::Mat* identifiedAreas, cv::Mat backgroundMask, int numberOfLowerDamageAreas, vector<ImageArea>* damagedAreas);
	void attachClosestLowerDamageAreaToOneArea(int area_index, int block_x, int block_y, cv::Mat* mat, cv::Mat* identified_areas_after_iteration,
		const cv::Mat& background_mask, int* number_of_lower_damage_areas, cv::Mat* processed_damaged_blocks, vector<ImageArea>* damagedAreas);
	bool blockWasChanged(const cv::Mat& former, const cv::Mat& updated, int x, int y);

	static cv::Mat drawHighDamageAreasFromPreview(Image* image);
};

