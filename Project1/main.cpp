#include "main.h"
#include "ProcessingPipeline.h"

using namespace std;

int main()
{
	Image* image;
	auto processingPipeline = new ProcessingPipeline();

	int images = 9;

    for(int i = 1; i <= images; i++)
    {
		string imageName = "Images/";
		imageName.append(std::to_string(i));
		imageName.append(".bmp");

		try {
			image = new Image(imread(imageName, cv::IMREAD_GRAYSCALE));
			processingPipeline->processImage(image);
			cv::waitKey();
		}
        catch(...)
        {
        }
    }

	cv::waitKey(100000);
    return 0;
}
