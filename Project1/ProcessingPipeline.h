#pragma once

#include <opencv2/highgui.hpp>
#include "Preprocessor.h"
#include "BackgroundSubstractor.h"
#include "OrientationsEstimator.h"
#include "FrequencyEstimator.h"
#include "DamageDetector.h"
#include "GaborFilter.h"
#include "Image.h"

class ProcessingPipeline {
public:
	ProcessingPipeline();
    void showProcessSteps(Image* image);
    void processImage(Image* image);
};

