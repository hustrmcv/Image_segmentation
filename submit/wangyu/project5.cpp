
#include<opencv2\opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

void aveMoveThreshold(Mat &srcImage, Mat &dstImage, int num);

int main(void)
{
	Mat srcImage1 = imread("Fig1049(a)(spot_shaded_text_image).tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat srcImage2 = imread("Fig1050(a)(sine_shaded_text_image).tif", CV_LOAD_IMAGE_GRAYSCALE);
	if (!srcImage1.data) { cout << "图像1打开失败" << endl; return -1; }
	if (!srcImage2.data) { cout << "图像2打开失败" << endl; return -1; }
	imshow("原始图1", srcImage1);
	imshow("原始图2", srcImage2);

	Mat dstImage1;
	dstImage1.create(srcImage1.size(), srcImage1.type());
	aveMoveThreshold(srcImage1, dstImage1, 20);
	imshow("效果图1", dstImage1);

	Mat dstImage2;
	dstImage2.create(srcImage2.size(), srcImage2.type());
	aveMoveThreshold(srcImage2, dstImage2, 20);
	imshow("效果图2", dstImage2);

	waitKey(0);
}

//移动平均阈值分割
void aveMoveThreshold(Mat & srcImage, Mat &dstImage, int num)
{
	double ret;			//用于计算1/n(Zk+1-Zk-n)；
	uchar *p = srcImage.data, *dp = dstImage.data;
	double mpre = p[0] / num, mnow;
	for (int i = 0; i != srcImage.rows; ++i)
	{
		for (int j = 0; j != srcImage.cols; ++j)
		{
			int count = srcImage.cols*i + j;
			if (count < num + 1)
				ret = p[count];
			else
				ret = p[count] - p[count - num - 1];
			ret = ret / num;
			mnow = mpre + ret;
			mpre = mnow;
			if (p[count] > (int)mpre/2)
				dp[count] = 255;
			else
				dp[count] = 0;
		}
	}
}
