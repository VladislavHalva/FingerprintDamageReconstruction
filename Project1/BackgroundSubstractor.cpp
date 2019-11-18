#include "BackgroundSubstractor.h"


BackgroundSubstractor::BackgroundSubstractor()
{
}


void BackgroundSubstractor::correctInnerBlocksEstimatedAsBackground(Image* image)
{
    cv::Mat backgroundMask = image->getBackgroundMask();
    cv::Mat validityMask = cv::Mat::zeros(backgroundMask.size(), CV_8U);

    for (int blockX = 0; blockX < backgroundMask.cols; blockX++)
    {
        for (int blockY = 0; blockY < backgroundMask.rows; blockY++)
        {
            //is bg and border ==> valid
            if (Image::isElementBorderElementOfMat(blockX, blockY, backgroundMask)
                && (int)backgroundMask.at<unsigned char>(blockY, blockX) == BACKGROUND)
            {
                validityMask.at<unsigned char>(blockY, blockX) = VALID;
            }
            //not border ==> has to have valid bg neigh
            else if ((int)backgroundMask.at<unsigned char>(blockY, blockX) == BACKGROUND)
            {
				if(hasBackgroundNeighbor(blockX, blockY, backgroundMask, validityMask))
				{
					validityMask.at<unsigned char>(blockY, blockX) = VALID;
				}
            }
        }
    }

    for (int blockX = 0; blockX < backgroundMask.cols; blockX++)
    {
        for (int blockY = 0; blockY < backgroundMask.rows; blockY++)
        {
            if ((int)backgroundMask.at<unsigned char>(blockY, blockX) == BACKGROUND
                && (int)validityMask.at<unsigned char>(blockY, blockX) == NOT_VALID)
            {
                backgroundMask.at<unsigned char>(blockY, blockX) = FOREGROUND;
            }
        }
    }
}


void BackgroundSubstractor::estimateBackgroundAreaFromVariance(Image* image)
{
    cv::Mat img = image->getProcessedImage();
    int blockSize = image->getBlockSize();
    cv::Mat backgroundMask = cv::Mat::zeros(img.rows / blockSize, img.cols / blockSize, CV_8U);

    for (int blockX = 0; blockX < backgroundMask.cols; blockX++)
    {
        for (int blockY = 0; blockY < backgroundMask.rows; blockY++)
        {
            //calculate variance in block
            cv::Scalar meanSc, devSc;
            double mean, stdDev, variance;

            //subduct only the current block
            cv::Mat block = img(cv::Rect(blockX * blockSize, blockY * blockSize, blockSize, blockSize));
            meanStdDev(block, meanSc, devSc);
            mean = meanSc.val[0];
            stdDev = devSc.val[0];
            variance = pow(stdDev, 2);

            //estimate bg by gray intensity variance in block
            if (variance < this->backgroundTreshold)
            {
                backgroundMask.at<unsigned char>(blockY, blockX) = BACKGROUND;
            }

            //border blocks have higher probability to be bg
            if (Image::isElementBorderElementOfMat(blockX, blockY, backgroundMask))
            {
                if (variance < this->backgroundTreshold)
                {
                    backgroundMask.at<unsigned char>(blockY, blockX) = BACKGROUND;
                }
            }
        }
    }

    image->setBackgroundMask(backgroundMask);
    interpolateBackgroundMask(image, 3);
    correctInnerBlocksEstimatedAsBackground(image);
}


void BackgroundSubstractor::interpolateBackgroundMask(Image* image, int cycles)
{
    cv::Mat backgroundMask = image->getBackgroundMask();

    //number of interpolation cycles 
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        //all blocks (i,j)
        for (int i = 0; i < backgroundMask.cols; i++)
        {
            for (int j = 0; j < backgroundMask.rows; j++)
            {
                int bgBlocksInNeigh = 0;

                //neighbourhood blocks (i+u,j+v)
                for (int u = -1; u <= 1; u++)
                {
                    for (int v = -1; v <= 1; v++)
                    {
                        if (Image::isElementInMatSizeRange(i + u, j + v, backgroundMask))
                        {
                            if ((int)backgroundMask.at<unsigned char>(j + v, i + u) == BACKGROUND)
                            {
                                bgBlocksInNeigh++;
                            }
                        }
                    }
                }

                if (Image::isElementBorderElementOfMat(i, j, backgroundMask))
                {
                    if (bgBlocksInNeigh > 1)
                    {
                        //is background too, on the border if both neigh blocks are bg
                        backgroundMask.at<unsigned char>(j, i) = BACKGROUND;
                    }
                }

                else if (bgBlocksInNeigh < 4)
                {
                    //is not background, too little neighbors are not background neither
                    backgroundMask.at<unsigned char>(j, i) = FOREGROUND;
                }

                else if (bgBlocksInNeigh > 4)
                {
                    //is background, most of neighbors are background either
                    backgroundMask.at<unsigned char>(j, i) = BACKGROUND;
                }
            }
        }
    }

    image->setBackgroundMask(backgroundMask);
}


bool BackgroundSubstractor::hasBackgroundNeighbor(int blockX, int blockY, const cv::Mat& backgroundMask)
{
	bool backgroundInNeigh = false;

	for (int x = blockX - 1; x <= blockX + 1; x++)
	{
		for (int y = blockY - 1; y <= blockY + 1; y++)
		{
			if (Image::isElementInMatSizeRange(x, y, backgroundMask))
			{
				if (backgroundMask.at<unsigned char>(y, x) == BACKGROUND)
					backgroundInNeigh = true;
			}
		}
	}

	return backgroundInNeigh;
}


bool BackgroundSubstractor::hasBackgroundNeighbor(int blockX, int blockY, const cv::Mat backgroundMask,
	const cv::Mat validityMask) {
	int validBlocksInNeigh = 0;

	//neighbourhood blocks (i+u,j+v)
	for (int u = -1; u <= 1; u++) {
		for (int v = -1; v <= 1; v++) {
			if (Image::isElementInMatSizeRange(blockX + u, blockY + v, backgroundMask)) {
				if ((int)validityMask.at<unsigned char>(blockY + v, blockX + u) == VALID) {
					validBlocksInNeigh++;
				}
			}
		}
	}

	return validBlocksInNeigh > 0;
}


cv::Mat BackgroundSubstractor::colorBackgroundAreasWhite(Image* image)
{
    cv::Mat finalImage;
    image->getProcessedImage().copyTo(finalImage);
    int blockSize = image->getBlockSize();

    for (int i = 0; i < finalImage.cols; i++)
    {
        for (int j = 0; j < finalImage.rows; j++)
        {
            if (image->getBackgroundMask().at<unsigned char>(j / blockSize, i / blockSize) == BACKGROUND)
            {
                finalImage.at<unsigned char>(j, i) = 255;
            }
        }
    }

    return finalImage;
}


cv::Mat BackgroundSubstractor::drawBackground(Image* image)
{
    cv::Mat finalImage;
    image->getProcessedImage().copyTo(finalImage);
    int blockSize = image->getBlockSize();

    for (int i = 0; i < finalImage.cols; i++)
    {
        for (int j = 0; j < finalImage.rows; j++)
        {
            if (image->getBackgroundMask().at<unsigned char>(j / blockSize, i / blockSize) == BACKGROUND)
            {
                finalImage.at<unsigned char>(j, i) = 0;
            }
			else
			{
				finalImage.at<unsigned char>(j, i) = 255;
			}
        }
    }

    return finalImage;
}


bool BackgroundSubstractor::isBackgroundBlock(int blockX, int blockY, cv::Mat backgroundMask)
{
    if ((int)backgroundMask.at<unsigned char>(blockY, blockX) == BACKGROUND)
    {
        return true;
    }
    return false;
}


bool BackgroundSubstractor::isBackgroundPixel(int pixelX, int pixelY, Image* image)
{
    cv::Mat backgroundMask = image->getBackgroundMask();
    int blockSize = image->getBlockSize();

    if ((int)backgroundMask.at<unsigned char>(pixelY / blockSize, pixelX / blockSize) == BACKGROUND)
    {
        return true;
    }
    return false;
}
