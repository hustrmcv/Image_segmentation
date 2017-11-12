//手动选择阈值进行分割
#include<opencv2\opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main(void)
{
	Mat srcImage=imread("Fig1001(a)(constant_gray_region).tif", CV_LOAD_IMAGE_GRAYSCALE); 

	if (!srcImage.data) { cout << "图片打开失败" << endl; return -1; }
	imshow("灰度图", srcImage);
	cout << srcImage.rows << " " << srcImage.cols << endl;
	//统计灰度级值，用于寻找合适阈值
	int table[256] = { 0 };
	for (size_t i = 0; i != srcImage.rows; ++i)
	{
		uchar *p = srcImage.ptr<uchar>(i);
		for (size_t j = 0; j != srcImage.cols; ++j)
		{
			table[p[j]]++;
		}
	}
	for (int i = 0; i < 256; ++i) {
		if (table[i])
			cout << "i: " << i << " " << "value: " << table[i] << endl;
	}
	//进行图像分割，手动填写阈值
	int threshold = 150;
	for (size_t i = 0; i != srcImage.rows; ++i)
	{
		uchar *p = srcImage.ptr<uchar>(i);
		for (size_t j = 0; j != srcImage.cols; ++j)
		{
			p[j] < threshold ? p[j] = 0 : p[j] = 255;
		}
	}
	imshow("效果图", srcImage);
	waitKey(0);
	return 0;
}
