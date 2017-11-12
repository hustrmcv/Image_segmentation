
//分块局部的阈值分割
#include<opencv2\opencv.hpp>
#include<iostream>
#include"myThreshold.h"
using  namespace std;
using namespace cv;

int main(void)
{
	Mat srcImage = imread("Fig1046(a)(septagon_noisy_shaded).tif",CV_LOAD_IMAGE_GRAYSCALE);
	if (!srcImage.data) { cout << "图片打开失败" << endl; return -1;}
	imshow("原图像", srcImage);
	VariableThresholdSegmentation(srcImage, 2, 3);
	imshow("效果图", srcImage);
	waitKey(0);
	return 0;
}
