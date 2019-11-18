#include "ProcessingPipeline.h"
#include "SingularityDetector.h"
#include "HighDamageDetector.h"

ProcessingPipeline::ProcessingPipeline() {
}


void ProcessingPipeline::processImage(Image* image)
{
	auto preproc = new Preprocessor();
	preproc->equalize(image);
	preproc->smoothenImage(image, 1);
	preproc->normalize(image);

	auto bSubstractor = new BackgroundSubstractor();
	bSubstractor->estimateBackgroundAreaFromVariance(image);

	auto oEstimator = new OrientationsEstimator();
	oEstimator->computeOrientationField(image);
	oEstimator->smoothenOrientationField(oEstimator->getThetaX(), oEstimator->getThetaY(), image);

	auto fEstimator = new FrequencyEstimator();
	fEstimator->computeFrequencyField(image);
	fEstimator->smoothenFrequencyField(image);

	auto damageDetector = new DamageDetector();
	damageDetector->setup(image);
	damageDetector->detectDamagedAreas();

	auto singularityDetector = new SingularityDetector();
	singularityDetector->findSingularities(image);
	singularityDetector->markDamageAreasThatContainCoreOrDelta(image);

	oEstimator->updateOrientationsBasedOnDamage(image);
	oEstimator->smoothenOrientationField(image->thetaX, image->thetaY, image);

	auto gaborFilter = new GaborFilter();
	gaborFilter->setup(image);
	gaborFilter->filter();
	cv::Mat filteredImage = gaborFilter->getResultImage();
	image->setProcessedImage(filteredImage);
	
	if(DEBUG) showProcessSteps(image);
}


void ProcessingPipeline::showProcessSteps(Image* image) 
{
	cv::Mat bgImage = BackgroundSubstractor::drawBackground(image);
	cv::Mat oFieldImage = OrientationsEstimator::drawOrientationField(image, true);
	cv::Mat fFieldImage = FrequencyEstimator::drawFrequencyField(image);
    cv::Mat qualityImage = DamageDetector::drawQualityMap(image);
	cv::Mat singularityMap = SingularityDetector::drawSingularityMap(image);
	cv::Mat highDamageAreas = HighDamageDetector::drawHighDamageAreasFromPreview(image);

	cv::Mat firstRow;
	cv::hconcat(image->getImage(), bgImage, firstRow);
	cv::hconcat(firstRow, oFieldImage, firstRow);
	cv::hconcat(firstRow, fFieldImage, firstRow);

	cv::Mat secondRow;
	cv::hconcat(qualityImage, highDamageAreas, secondRow);
	cv::hconcat(secondRow, singularityMap, secondRow);
	cv::hconcat(secondRow, image->getProcessedImage(), secondRow);

	cv::Mat complete;
	cv::vconcat(firstRow, secondRow, complete);

	double scale = 600. / complete.rows;

	cv::resize(complete, complete, cv::Size(static_cast<int>(complete.cols * scale), 600));

	cv::imshow("Process steps", complete);
}