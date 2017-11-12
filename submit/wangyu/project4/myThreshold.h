
#pragma once
#ifndef MYTHRESHOLD_H
#define MYTHRESHOLD_H

#include<opencv2\opencv.hpp>
using namespace cv;

void VariableThresholdSegmentation(Mat &srcImage, int row, int col);
void getPartGrayValue(Mat &srcImage, double *value, int x, int y, int a1, int b1);
int OtsuThresh(double * value);
void partSegment(Mat &srcImage, int x, int y, int a1, int b1, int T);

#endif // !MYTHRESHOLD_H
