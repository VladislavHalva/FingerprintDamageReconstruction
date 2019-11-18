#pragma once

#include "Image.h"
class Preprocessor {
public:
	Preprocessor();
	void normalize(Image* image);
	void equalize(Image* image);
	void smoothenImage(Image* image, int sigma);
};

