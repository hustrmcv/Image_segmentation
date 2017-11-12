
//采用多阈值对图像进行分割；以下确定为两个阈值，未自动确定多阈值
#include<opencv2\opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

void OtsuMultiThresh(double * value, int &T1, int &T2);
void getGrayValue(Mat &srcImage, double *value);
void multiThresholdSegment(Mat &srcImage, int T1, int T2);

int main(void)
{
	Mat srcImage = imread("Fig1045(a)(iceberg).tif", CV_LOAD_IMAGE_GRAYSCALE);
	if (!srcImage.data) { cout << "图像打开失败" << endl; return -1; }
	imshow("原始图", srcImage);

	double value[256] = { 0 };
	getGrayValue(srcImage, value);

	int T1, T2;
	OtsuMultiThresh(value, T1, T2);

	multiThresholdSegment(srcImage, T1, T2);
	imshow("效果图", srcImage);
	waitKey(0);
	return 0;
}
//利用大津法的原理来寻找两个阈值
void OtsuMultiThresh(double * value, int &T1, int &T2)
{
	cout << "OK1" << endl;
	int L = 256;
	double p1, p2, p3, m1, m2, m3, mG, deltatmp, deltmax = 0;
	for (int count = 1; count < L-3; ++count)
	{
		int K1 = count;
		for (int K2 = K1 + 1; K2 < L - 2; ++K2)
		{
			p1 = p2 = p3 = m1 = m2 = m3 = 0;
			for (int i = 0; i < L; ++i)
			{
				if (i <= K1)
				{
					p1 += value[i];
					m1 += (i + 1)*value[i];
				}
				else if (K1 < i && i <= K2)
				{
					p2 += value[i];
					m2 += (i + 1)*value[i];
				}
				else
				{
					m3 += (i + 1)*value[i];
				}
			}
			p3 = 1 - p1 - p2;
			mG = m1 + m2 + m3;
			if (p1 != 0)
				m1 = m1 / p1;
			if (p2 != 0)
				m2 = m2 / p2;
			if (p3 != 0)
				m3 = m3 / p3;
			deltatmp = p1*(m1 - mG)*(m1 - mG) + p2*(m2 - mG)*(m2 - mG) + p3*(m3 - mG)*(m3 - mG);
			if (deltatmp > deltmax)
			{
				deltmax = deltatmp;
				T1 = K1;
				T2 = K2;
			}
		}
	}
}
//获得灰度值
void getGrayValue(Mat & srcImage, double * value)
{
	for (int i = 0; i != srcImage.rows; ++i)
	{
		uchar *p = srcImage.ptr<uchar>(i);
		for (int j = 0; j != srcImage.cols; ++j)
		{
			value[p[j]]++;
		}
	}
	int sum = srcImage.rows*srcImage.cols;
	for (int i = 0; i < 256; ++i)
		value[i] = value[i] / sum;
}

//分割
void multiThresholdSegment(Mat & srcImage, int T1, int T2)
{
	for (int i = 0; i != srcImage.rows; ++i)
	{
		uchar *p = srcImage.ptr<uchar>(i);
		for (int j = 0; j != srcImage.cols; ++j)
		{
			if (p[j] <= T1)
				p[j] = 0;
			else if (T1 < p[j] && p[j] <= T2)
				p[j] = 128;
			else
				p[j] = 255;
		}
	}
}
