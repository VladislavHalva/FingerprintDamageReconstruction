#include "HighDamageDetector.h"
#include "BackgroundSubstractor.h"
#include "FloodFill.h"


HighDamageDetector::HighDamageDetector()
{
}


void HighDamageDetector::findHeavilyDamagedAreas(Image* image)
{
	cv::Mat qualityMap = image->getQualityMap();
	cv::Mat backgroundMask = image->getBackgroundMask();

	//divide quality map to 3 quality categories
	cv::Mat qualityCategoryMap = divideQualityMapToCategories(qualityMap, backgroundMask);

	cv::Mat areasClasificationMap = findContinuousAreasInQualityMap(qualityCategoryMap, backgroundMask);

	vector<ImageArea> damagedAreas;
	cv::Mat identifiedAreas = identifyAreas(areasClasificationMap, backgroundMask, &damagedAreas);

	image->setHighlyDamagedAreas(damagedAreas);
	image->setHighlyDamagedAreasPreview(identifiedAreas);
}


cv::Mat HighDamageDetector::divideQualityMapToCategories(cv::Mat qualityMap, const cv::Mat& backgroundMask)
{
	cv::Mat qualityCategoryMap = cv::Mat::zeros(qualityMap.size(), CV_8U);

	//divide image blocks to low damaged, damaged and OK
	for (int blockX = 0; blockX < qualityMap.cols; blockX++) {
		for (int blockY = 0; blockY < qualityMap.rows; blockY++) {
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				continue;
			}

			if (qualityMap.at<double>(blockY, blockX) < 0.62) {
				qualityCategoryMap.at<unsigned char>(blockY, blockX) = LOW_DAMAGE;
			}

			if (qualityMap.at<double>(blockY, blockX) < 0.44) {
				qualityCategoryMap.at<unsigned char>(blockY, blockX) = DAMAGED;
			}
		}
	}

	return qualityCategoryMap;
}


cv::Mat HighDamageDetector::findContinuousAreasInQualityMap(cv::Mat qualityCategoryMap, const cv::Mat backgroundMask)
{
	cv::Mat classificationMap = cv::Mat::zeros(qualityCategoryMap.size(), CV_8U);

	for (int blockX = 0; blockX < qualityCategoryMap.cols; blockX++)
	{
		for (int blockY = 0; blockY < qualityCategoryMap.rows; blockY++) {
			int damagedInNeigh = 0;
			int lowDamagedInNeigh = 0;

			//how many neighborhood blocks are damaged
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					if (Image::isElementInMatSizeRange(blockX + u, blockY + v, qualityCategoryMap)) {
						if ((int)qualityCategoryMap.at<unsigned char>(blockY + v, blockX + u) == LOW_DAMAGE) {
							lowDamagedInNeigh++;
						}
						if ((int)qualityCategoryMap.at<unsigned char>(blockY + v, blockX + u) == DAMAGED) {
							damagedInNeigh++;
						}
					}
				}
			}

			double neighDamageIndex = damagedInNeigh + 0.6 * lowDamagedInNeigh;

			if (neighDamageIndex > 4)
			{
				classificationMap.at<unsigned char>(blockY, blockX) = BORDER_BLOCK;
			}

			if (neighDamageIndex > 7)
			{
				classificationMap.at<unsigned char>(blockY, blockX) = INNER_BLOCK;
			}
		}
	}

	return classificationMap;
}


cv::Mat HighDamageDetector::identifyAreas(cv::Mat areasClasificationMap, cv::Mat backgroundMask, vector<ImageArea>* damagedAreas)
{
	cv::Mat identifiedAreas;
	areasClasificationMap.copyTo(identifiedAreas);
	identifiedAreas.convertTo(identifiedAreas, CV_32S);
	int areaIndex = 1;

	int numberOfLowerDamageAreas = convertClassificationValueToExt(&identifiedAreas);

	removeLowerDamageAreasNotConnectedToHiglyDamagedArea(&identifiedAreas, backgroundMask, &numberOfLowerDamageAreas);

	*damagedAreas = fillHighlyDamagedAreas(&identifiedAreas, backgroundMask, &areaIndex);
	int numberOfAreas = areaIndex - 1;

	attachLowerDamageAreas(&identifiedAreas, backgroundMask, numberOfLowerDamageAreas, damagedAreas);

	return identifiedAreas;
}


int HighDamageDetector::convertClassificationValueToExt(cv::Mat* identifiedAreas) {
	int numberOfLowerDamageAreas = 0;

	for (int blockX = 0; blockX < identifiedAreas->cols; blockX++) {
		for (int blockY = 0; blockY < identifiedAreas->rows; blockY++) {
			if (identifiedAreas->at<int>(blockY, blockX) == BORDER_BLOCK) {
				identifiedAreas->at<int>(blockY, blockX) = LOW_DAMAGE_EXT;
				numberOfLowerDamageAreas++;
			}
			if (identifiedAreas->at<int>(blockY, blockX) == INNER_BLOCK) {
				identifiedAreas->at<int>(blockY, blockX) = DAMAGED_EXT;
			}
		}
	}

	return numberOfLowerDamageAreas;
}


void HighDamageDetector::removeLowerDamageAreasNotConnectedToHiglyDamagedArea(cv::Mat* identifiedAreas,
	const cv::Mat& backgroundMask, int* numberOfLowerDamageAreas)
{
	cv::Mat identifiedAreasCopy;
	identifiedAreas->copyTo(identifiedAreasCopy);

	for (int blockX = 0; blockX < identifiedAreasCopy.cols; blockX++) {
		for (int blockY = 0; blockY < identifiedAreasCopy.rows; blockY++) {
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				continue;
			}

			if (identifiedAreasCopy.at<int>(blockY, blockX) == LOW_DAMAGE_EXT) {
				vector<cv::Point> allNeighboringLowDamage;
				bool neighboringHighDamageBlock = FloodFill::searchFloodFillStep(blockX, blockY, &identifiedAreasCopy, LOW_DAMAGE_EXT,
					0, DAMAGED_EXT, &allNeighboringLowDamage);

				//remove low damage blocks, that have no connection to a highly damaged block
				if (!neighboringHighDamageBlock) {
					for (int i = 0; i < allNeighboringLowDamage.size(); i++) {
						identifiedAreas->at<int>(allNeighboringLowDamage.at(i)) = 0;
						*numberOfLowerDamageAreas = *numberOfLowerDamageAreas - 1;
					}
				}
			}
		}
	}
}


vector<ImageArea> HighDamageDetector::fillHighlyDamagedAreas(cv::Mat* identifiedAreas, const cv::Mat& backgroundMask, int* areaIndex)
{
	vector<ImageArea> damagedAreas;

	for (int blockX = 0; blockX < identifiedAreas->cols; blockX++) {
		for (int blockY = 0; blockY < identifiedAreas->rows; blockY++) {
			//skip background
			if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask)) {
				continue;
			}

			//fill damaged area
			if (identifiedAreas->at<int>(blockY, blockX) == DAMAGED_EXT) {
				vector<cv::Point> areaBlocks;
				FloodFill::floodFillStep(blockX, blockY, identifiedAreas, DAMAGED_EXT, *areaIndex, &areaBlocks);
				damagedAreas.push_back(ImageArea(areaBlocks, DAMAGED));
				*areaIndex = *areaIndex + 1;
			}
		}
	}

	return damagedAreas;
}


void HighDamageDetector::attachLowerDamageAreas(cv::Mat* identifiedAreas, cv::Mat backgroundMask, int numberOfLowerDamageAreas, vector<ImageArea>* damagedAreas)
{
	cv::Mat processedDamagedBlocks = cv::Mat::zeros(identifiedAreas->size(), CV_32S);

	cv::Mat identifiedAreasBeforeIteration;
	cv::Mat identifiedAreasAfterIteration;
	identifiedAreas->copyTo(identifiedAreasBeforeIteration);
	identifiedAreas->copyTo(identifiedAreasAfterIteration);

	while (numberOfLowerDamageAreas > 0) {

		for (int blockX = 0; blockX < identifiedAreas->cols; blockX++)
		{
			for (int blockY = 0; blockY < identifiedAreas->rows; blockY++)
			{
				if (BackgroundSubstractor::isBackgroundBlock(blockX, blockY, backgroundMask))
					continue;
				if (processedDamagedBlocks.at<int>(blockY, blockX) == CHECKED)
					continue;

				//high damage area
				if (identifiedAreasBeforeIteration.at<int>(blockY, blockX) != 0 &&
					identifiedAreasBeforeIteration.at<int>(blockY, blockX) != LOW_DAMAGE_EXT)
				{
					int areaIndex = identifiedAreasBeforeIteration.at<int>(blockY, blockX);

					attachClosestLowerDamageAreaToOneArea(areaIndex, blockX, blockY, &identifiedAreasBeforeIteration, &identifiedAreasAfterIteration,
						backgroundMask, &numberOfLowerDamageAreas, &processedDamagedBlocks, damagedAreas);

				}
			}
		}
		//update damage map for next iteration of damage extension
		identifiedAreasAfterIteration.copyTo(identifiedAreasBeforeIteration);
	}
	identifiedAreasBeforeIteration.copyTo(*identifiedAreas);
}


void HighDamageDetector::attachClosestLowerDamageAreaToOneArea(int areaIndex, int blockX, int blockY, cv::Mat* identifiedAreasBeforeIteration,
	cv::Mat* identifiedAreasAfterIteration, const cv::Mat& backgroundMask, int* numberOfLowerDamageAreas,
	cv::Mat* processedDamagedBlocks, vector<ImageArea>* damagedAreas)
{
	for (int damageBlockX = blockX; damageBlockX < identifiedAreasBeforeIteration->cols; damageBlockX++)
	{
		for (int damageBlockY = 0; damageBlockY < identifiedAreasBeforeIteration->rows; damageBlockY++)
		{
			if (BackgroundSubstractor::isBackgroundBlock(damageBlockX, damageBlockY, backgroundMask))
				continue;

			//for each block of the current damage area
			if (identifiedAreasBeforeIteration->at<int>(damageBlockY, damageBlockX) == areaIndex)
			{
				for (int maskX = -1; maskX <= 1; maskX++) {
					for (int maskY = -1; maskY <= 1; maskY++) {
						if (!Image::isElementInMatSizeRange(damageBlockX + maskX, damageBlockY + maskY, *identifiedAreasBeforeIteration))
							continue;
						if (BackgroundSubstractor::isBackgroundBlock(damageBlockX + maskX, damageBlockY + maskY, backgroundMask))
							continue;

						//low damage block in neighborhood, but was not already connected to damaged area in this iteration 
						if (identifiedAreasBeforeIteration->at<int>(damageBlockY + maskY, damageBlockX + maskX) == LOW_DAMAGE_EXT &&
							!blockWasChanged(*identifiedAreasBeforeIteration, *identifiedAreasAfterIteration, damageBlockX + maskX, damageBlockY + maskY)) {

							//add block to damage area in map
							identifiedAreasAfterIteration->at<int>(damageBlockY + maskY, damageBlockX + maskX) = areaIndex;
							//add block to damaged area structure either
							damagedAreas->at(areaIndex - 1).addPoint(cv::Point(damageBlockX + maskX, damageBlockY + maskY));

							*numberOfLowerDamageAreas = *numberOfLowerDamageAreas - 1;
						}
					}
				}
				processedDamagedBlocks->at<int>(damageBlockY, damageBlockX) = CHECKED;
			}
		}
	}
}


bool HighDamageDetector::blockWasChanged(const cv::Mat& former, const cv::Mat& updated, int x, int y)
{
	if (former.at<int>(y, x) == updated.at<int>(y, x))
		return false;
	else
		return true;
}

cv::Mat HighDamageDetector::drawHighDamageAreasFromPreview(Image* image)
{
	cv::Mat preview = image->getHighlyDamagedAreasPreview();

	double min, max;
	cv::minMaxLoc(preview, &min, &max);

	cv::Mat identAreasShow;
	preview.convertTo(identAreasShow, CV_8U, 255.0 / (max - min), min * 255.0 / (min - max));
	identAreasShow = Image::extendBlocksToFullSizeImage(identAreasShow, image->getBlockSize(), image->getImage().size());

	return identAreasShow;
}
