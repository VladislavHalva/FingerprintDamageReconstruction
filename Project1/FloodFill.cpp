#include "FloodFill.h"
#include "Image.h"


FloodFill::FloodFill() {
}

void FloodFill::floodFillStep(int x, int y, cv::Mat* mat, int oldValue, int newValue, std::vector<cv::Point>* areaBlocks) {
	std::vector<cv::Point> stack;

	stack.push_back(cv::Point(x, y));

	while (!stack.empty()) {
		if (Image::isElementInMatSizeRange(stack.back().x, stack.back().y, *mat)) {
			if ((*mat).at<int>(stack.back()) == oldValue) {
				(*mat).at<int>(stack.back()) = newValue;

				int bX = stack.back().x;
				int bY = stack.back().y;

				areaBlocks->push_back(cv::Point(bX, bY));
				stack.pop_back();

                //4-way
				stack.push_back(cv::Point(bX + 1, bY));
				stack.push_back(cv::Point(bX - 1, bY));
				stack.push_back(cv::Point(bX, bY + 1));
				stack.push_back(cv::Point(bX, bY - 1));

                //extension to 8-way
				stack.push_back(cv::Point(bX + 1, bY + 1));
				stack.push_back(cv::Point(bX + 1, bY - 1));
				stack.push_back(cv::Point(bX - 1, bY + 1));
				stack.push_back(cv::Point(bX - 1, bY - 1));
			}
			else {
				//different color
				stack.pop_back();
			}
		}
		else {
			//out of range
			stack.pop_back();
		}
	}
}


bool FloodFill::searchFloodFillStep(int x, int y, cv::Mat* mat, int oldValue, int newValue, int searchedValue, vector<cv::Point>* allNeighboring) {
	std::vector<cv::Point> stack;
	bool foundSearchedNeighbor = false;

	stack.push_back(cv::Point(x, y));

	while (!stack.empty()) {
		if (Image::isElementInMatSizeRange(stack.back().x, stack.back().y, *mat)) {
			if ((*mat).at<int>(stack.back()) == oldValue) {
				(*mat).at<int>(stack.back()) = newValue;

				int bX = stack.back().x;
				int bY = stack.back().y;

				allNeighboring->push_back(cv::Point(bX, bY));
				stack.pop_back();

				stack.push_back(cv::Point(bX + 1, bY));
				stack.push_back(cv::Point(bX - 1, bY));
				stack.push_back(cv::Point(bX, bY + 1));
				stack.push_back(cv::Point(bX, bY - 1));
			}
			else if ((*mat).at<int>(stack.back()) == searchedValue) {
				foundSearchedNeighbor = true;
				stack.pop_back();
			}
			else {
				//different color
				stack.pop_back();
			}
		}
		else {
			//out of range
			stack.pop_back();
		}
	}

	return foundSearchedNeighbor;
}