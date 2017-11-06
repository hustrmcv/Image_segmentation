#include <iostream>
#include "opencv.hpp"
#include "highgui.h"
#include "imgproc.hpp"
//手动选择阈值进行二值化
using namespace cv;
using namespace std;
int threshold_type=1;
int threshold_value=150,size_=25,length_min=100;
int compensation=50,judge=0.95;
Mat src,src_binary,src_gray,src_edge,dstIMAGE;
double getDistance (CvPoint pointO,CvPoint pointA )
{
    double distance;
    distance = powf((pointO.x - pointA.x),2) + powf((pointO.y - pointA.y),2);
    distance = sqrtf(distance);

    return distance;
}
void Thres_hold(int, void*)
{
      int i,j;
      //图像灰度化
      cvtColor(src,src_binary,CV_BGR2GRAY);
      for(i=0;i<src_binary.rows;++i)
      for(j=0;j<src_binary.cols;++j)
      {
      //图像二值化
      if(src_binary.at<uchar>(i,j)>=threshold_value) src_binary.at<uchar>(i,j)=255;
      else src_binary.at<uchar>(i,j)=0;
      }
      imshow("二值化后的图像",src_binary);
}

//大津法寻找合适的阈值,并二值化
void find_binary_value(void)
{
      int pixel[255],i,j,t_max=0,t,state=0;
      for(i=0;i<254;i++) pixel[i]=0;
      float p[255],ut=0;
      cvtColor(src,src_binary,CV_BGR2GRAY);
      for(i=0;i<src_binary.rows;++i)
            for(j=0;j<src_binary.cols;++j)
                  pixel[src_binary.at<uchar>(i,j)]++;
      for(i=0;i<254;i++) p[i]=((float)pixel[i])/(src_binary.rows*src_binary.cols);
      for(i=0;i<254;i++) ut+=p[i]*i;
      float u1,u0,p1,p2,M1,M2,m1,m2;
      double g,g_max=0;

      for(t=0;t<255;t++)
      {
            p1=p2=0;
            M1=M2=0;
            cvtColor(src,src_binary,CV_BGR2GRAY);
            for(i=0;i<t;i++){
                  p1+=p[i];
                  M1+=i*p[i];
            }
            for(j=t;j<255;j++){
                  p2+=p[j];
                  M2+=j*p[j];
            }
            m1=M1/p1;
            m2=M2/p2;
            g=(p1*(m1-ut)*(m1-ut))+(p2*(m2-ut)*(m2-ut));
            if(g>g_max){t_max=t,g_max=g;}
      }
      for(i=0;i<src_binary.rows;++i)
      for(j=0;j<src_binary.cols;++j)
      {
      //图像二值化
      if(src_binary.at<uchar>(i,j)>=t_max) src_binary.at<uchar>(i,j)=255;
      else src_binary.at<uchar>(i,j)=0;
      }
      imshow("二值化后的图像",src_binary);

}
//根据尺寸区域中所有点的像素平均值确定区域阈值的函数(效果不佳,运算有点慢）

void part_adapt_binary_0(void)
{
            int i,j,size_0=7,x,y;
            long long part_sum_0=0;
            cvtColor(src,src_binary,CV_BGR2GRAY);
            blur(src_binary,src_binary,Size(3,3));
            for(i=0;i<src_binary.rows;++i)
            for(j=0;j<src_binary.cols;++j)
            {
                  part_sum_0=0;
                  int col,row;
                  for(row=-size_0/2;row<=size_0/2;row++)
                  for(col=-size_0/2;col<=size_0/2;col++){
                        if(i-row<0) x=0;else if(i-row>src_binary.rows-1) x=src_binary.rows-1;else x=i-row;
                        if(j-col<0) y=0;else if(i-row>src_binary.cols-1) x=src_binary.cols-1;else y=j-col;
                        part_sum_0=part_sum_0+((src_binary.at<uchar>(x,y)));
                  }
                        if(((long long)((src_binary.at<uchar>(i,j))*(size_0*size_0)))>=part_sum_0)
                        src_binary.at<uchar>(i,j)=255;
                        else src_binary.at<uchar>(i,j)=0;
      }
            imshow("二值化后的图像",src_binary);
}

//优化后的局部自适应寻找阈值并二值化（具有一定的泛适性）
void part_adapt_binary(int,void*)
{
            int i,j,x1,x2,y1,y2,area=0;
            long long part_sum=0;
            cvtColor(src,src_binary,CV_BGR2GRAY);
            //声明一个二维数组来存储局部像素值
            long long **Argv;
            Argv=new long long*[src_binary.rows];
            for(int ii=0;ii<src_binary.rows;ii++)
            {
                  Argv[ii]=new long long[src_binary.cols];
            }
            //一列一列的存储局部像素值
            for( i=0;i<src_binary.cols;i++)
            {
                  part_sum=0;
                  for( j=0;j<src_binary.rows;j++)
                  {
                        part_sum+=src_binary.at<uchar>(j,i);
                        if(i==0)
                              Argv[j][i]=part_sum;
                        else
                              Argv[j][i]=Argv[j][i-1]+part_sum;
                  }
            }
            //依据以（i，j）为中心的局部区域四个顶点的局部总像素值来进行二值化
            for(i=0;i<src_binary.cols;++i)
            for(j=0;j<src_binary.rows;++j)
            {
                  x1=i-size_/2;
                  x2=i+size_/2;
                  y1=j-size_/2;
                  y2=j+size_/2;
                  if(x1<0)
                  x1=0;
                  if(x2>=src_binary.cols)
                  x2=src_binary.cols-1;
                  if(y1<0)
                  y1=0;
                  if(y2>=src_binary.rows)
                  y2=src_binary.rows-1;
                  area=(x2-x1)*(y2-y1);
                  part_sum=Argv[y2][x2]-Argv[y1][x2]-Argv[y2][x1]+Argv[y1][x1];
                  if((long long)(src_binary.at<uchar>(j,i)*area)<((long long)part_sum*(100-compensation)/100))
                  src_binary.at<uchar>(j,i)=0;
                  else src_binary.at<uchar>(j,i)=255;

      }
            imshow("二值化后的图像",src_binary);
            //施放存储局部像素值的二维数组的内存
            for (i=0;i < src_binary.rows; ++i)
            {
            delete [] Argv[i];
            }
            delete [] Argv;
}
//基于Otsu的多阈值图像分割(暂时三阈值)
void multiple_binary(void)
{
      int pixel[255],i,j,t_min=0,t_max=255,t,t_max_choosed,state=0,sort_changed=0;
      vector<int> t_choosed;
      for(i=0;i<254;i++) pixel[i]=0;
      float p[255],ut=0,vt=0,judge=0,m1,m2,M1,M2,p1,p2;
      cvtColor(src,src_binary,CV_BGR2GRAY);
      for(i=0;i<src_binary.rows;++i)
            for(j=0;j<src_binary.cols;++j)
                  pixel[src_binary.at<uchar>(i,j)]++;
      for(i=0;i<254;i++) p[i]=((float)pixel[i])/(src_binary.rows*src_binary.cols);
      for(i=0;i<254;i++) ut+=p[i]*i;
      for(i=0;i<254;i++) vt+=(i-ut)*(i-ut)*p[i];
      while(t_choosed.size()<2)
      {
            double g,g_max;
            if(state==0){
                  g_max=0;
            for(t=0;t<255;t++)
            {
                  p1=p2=0;
                  M1=M2=0;
                  cvtColor(src,src_binary,CV_BGR2GRAY);
                  for(i=0;i<t;i++){
                        p1+=p[i];
                        M1+=i*p[i];
                  }
                  for(j=t;j<255;j++){
                        p2+=p[j];
                        M2+=j*p[j];
                  }
                  m1=M1/p1;
                  m2=M2/p2;
                  g=(p1*(m1-ut)*(m1-ut))+(p2*(m2-ut)*(m2-ut));
                  if(g>g_max){t_max_choosed=t,g_max=g;}
            }
            t_choosed.push_back(t_max_choosed);
            state++;
            }
            else {            //寻找最大的类间方差
                  g_max=0;
                  int t_max_q[t_choosed.size()+1];
                  double g_max_q[t_choosed.size()+1];
                  t_min=0,t_max=t_choosed[0]-1;
                  for(t=t_min;t<t_max;t++)
            {
                  p1=p2=0;
                  M1=M2=0;
                  cvtColor(src,src_binary,CV_BGR2GRAY);
                  for(i=t_min;i<=t;i++){
                        p1+=p[i];
                        M1+=i*p[i];
                  }
                  for(j=t+1;j<t_max;j++){
                        p2+=p[j];
                        M2+=j*p[j];
                  }
                  m1=M1/p1;
                  m2=M2/p2;
                  g=(p1*(m1-ut)*(m1-ut))+(p2*(m2-ut)*(m2-ut));
                  if(g>g_max) {t_max_choosed=t,g_max=g;}
            }

                  g_max_q[0]=g_max,t_max_q[0]=t_max_choosed;
                  g_max=0;
                  t_min=t_choosed[0]+1,t_max=255;
                  for(t=t_min;t<t_max;t++)
            {
                  p1=p2=0;
                  M1=M2=0;
                  cvtColor(src,src_binary,CV_BGR2GRAY);
                  for(i=t_min;i<=t;i++){
                        p1+=p[i];
                        M1+=i*p[i];
                  }
                  for(j=t+1;j<t_max;j++){
                        p2+=p[j];
                        M2+=j*p[j];
                  }
                  m1=M1/p1;
                  m2=M2/p2;
                  g=(p1*(m1-ut)*(m1-ut))+(p2*(m2-ut)*(m2-ut));
                  if(g>g_max){t_max_choosed=t,g_max=g;}
            }
                  g_max_q[1]=g_max,t_max_q[1]=t_max_choosed;
                  //对类内方差较大的一个类求其最适阈值
                  if(g_max_q[0]>g_max_q[1])
                  t_choosed.push_back(t_max_q[0]);
                  else t_choosed.push_back(t_max_q[1]);
      }
   }

      do{
                  //冒泡排序对求得的阈值进行从小到大排序
            sort_changed=0;
            int save_d;
            for(i=0;i<t_choosed.size()-1;i++){
                  if(t_choosed[i]>t_choosed[i+1]) {
                        save_d=t_choosed[i];
                        t_choosed[i]=t_choosed[i+1];
                        t_choosed[i+1]=save_d;
                        sort_changed++;
                  }
      }
      }while(sort_changed);

      cout<<t_choosed[0]<<'\t'<<t_choosed[1]<<endl;
      for(i=0;i<src_binary.rows;++i)
      for(j=0;j<src_binary.cols;++j)
      {
      //图像二值化
      if(src_binary.at<uchar>(i,j)<t_choosed[0]) src_binary.at<uchar>(i,j)=255;
      else if(src_binary.at<uchar>(i,j)>t_choosed[1]) src_binary.at<uchar>(i,j)=0;
      else src_binary.at<uchar>(i,j)=127;
      }
      imshow("二值化后的图像",src_binary);
}


//真·基于Otsu的多阈值图像分割
void multiple_binary_real(void)
{
      int pixel[255],i,j,k,t_min=0,t_max=255,t,t_max_choosed,state=0,sort_changed=0;      //
      vector<int> t_choosed;//存储筛选出的阈值
      vector<float> w_cluster,u_cluster;  //用来计算全局类间方差以判断是否继续阈值筛选
      for(i=0;i<254;i++) pixel[i]=0;      //存储像素点的分布
      float p[255],ut=0,vt=0,judge=0,m1,m2,M1,M2,p1,p2,SF=0,w,u,p_BC;   //p为存储各像素值所占比例，后面一系列变量均与阈值筛选相关
      cvtColor(src,src_binary,CV_BGR2GRAY);
      for(i=0;i<src_binary.rows;++i)
            for(j=0;j<src_binary.cols;++j)
                  pixel[src_binary.at<uchar>(i,j)]++;
      for(i=0;i<254;i++) p[i]=((float)pixel[i])/(src_binary.rows*src_binary.cols);
      for(i=0;i<254;i++) ut+=p[i]*i;
      for(i=0;i<254;i++) vt+=(i-ut)*(i-ut)*p[i];
      while(SF<0.9) // SF取值为0~1，用于判断是否退出阈值筛选，后面系数可调
      {
            w=u=0;
            double g,g_max;
            if(state==0){     //第一次筛选阈值
                  g_max=0;
            for(t=0;t<255;t++)
            {
                  p1=p2=0;
                  M1=M2=0;
                  cvtColor(src,src_binary,CV_BGR2GRAY);
                  for(i=0;i<t;i++){
                        p1+=p[i];
                        M1+=i*p[i];
                  }
                  for(j=t;j<255;j++){
                        p2+=p[j];
                        M2+=j*p[j];
                  }
                  m1=M1/p1;
                  m2=M2/p2;
                  g=(p1*(m1-ut)*(m1-ut))+(p2*(m2-ut)*(m2-ut));
                  if(g>g_max){t_max_choosed=t,g_max=g;}
            }
            t_choosed.push_back(0);
            t_choosed.push_back(t_max_choosed);
            t_choosed.push_back(255);
            state++;
            }
            else {            //寻找最大的类间方差，从而筛选出对应的阈值
                  g_max=0;
                  for(int r=0;r<t_choosed.size()-1;r++)
                  {
                        t_min=t_choosed[r],t_max=t_choosed[r+1];
                        for(t=t_min;t<t_max;t++)
                  {
                        p1=p2=0;
                        M1=M2=0;
                        cvtColor(src,src_binary,CV_BGR2GRAY);
                        for(i=t_min;i<=t;i++){
                              p1+=p[i];
                              M1+=i*p[i];
                        }
                        for(j=t+1;j<t_max;j++){
                              p2+=p[j];
                              M2+=j*p[j];
                        }
                        m1=M1/p1;
                        m2=M2/p2;
                        g=(p1*(m1-ut)*(m1-ut))+(p2*(m2-ut)*(m2-ut));
                        if(g>g_max) {t_max_choosed=t,g_max=g;}
                  }
                  }
                  t_choosed.push_back(t_max_choosed);
                  do{
                        //冒泡排序对求得的阈值进行从小到大排序
                  sort_changed=0;
                  int save_d;
                  for(i=0;i<t_choosed.size()-1;i++){
                        if(t_choosed[i]>t_choosed[i+1]) {
                              save_d=t_choosed[i];
                              t_choosed[i]=t_choosed[i+1];
                              t_choosed[i+1]=save_d;
                              sort_changed++;
                        }
                  }
                  }while(sort_changed);
      }
      //计算全局类间方差判断是否退出筛选阈值
      for(i=0;i<t_choosed.size()-1;i++){
            for(j=t_choosed[i];j<t_choosed[i+1];j++){
                  w+=p[j];
                  u+=p[j]*j;
            }
            if(!w_cluster.size()) w_cluster.push_back(w);
            else w_cluster.push_back(w-w_cluster[i-1]) ;
            if(!u_cluster.size()) u_cluster.push_back(u/w);
            else u_cluster.push_back(u/w-u_cluster[i-1]);
      }
      for(k=0;k<u_cluster.size();k++)
      p_BC+=w_cluster[k]*(u_cluster[k]/w_cluster[k]-ut)*(u_cluster[k]/w_cluster[k]-ut);
      SF=p_BC/vt;
   }
      cout<<t_choosed.size()<<endl;
      //依据选取的阈值，构造对应的阈值转换数组
      float p_binary[t_choosed.size()-1];
      int if_sep=0;
      for(i=0;i<t_choosed.size()-1;i++){
            p_binary[i]=i*(255/(t_choosed.size()-2));
      }

      //图像二值化
      for(i=0;i<src_binary.rows;++i)
      for(j=0;j<src_binary.cols;++j)
      {
            if_sep=0;
      for(k=0;k<t_choosed.size()-1&&!if_sep;k++)
      if(t_choosed[k]<src_binary.at<uchar>(i,j)&&src_binary.at<uchar>(i,j)<t_choosed[k+1])
      {
            if_sep=1;
            src_binary.at<uchar>(i,j)=p_binary[k];
      }
      }
      imshow("二值化后的图像",src_binary);
}

//纸张分割
void paper_segment(int,void *){
      vector <Vec4i> lines,hierarchy;
      vector<Vec4i> lines1,lines2;  //依据斜率将直线分成两部分存储
      vector<vector<Point> > contours;
      vector<Point> points;
      int i,j,x,y;
      cvtColor(src,src_gray,CV_BGR2GRAY);
      blur( src_gray,src_gray, Size(3,3));
      Canny(src_gray,src_edge,150,200,3); //利用CANNY算子提取图片轮廓信息
      findContours(src_edge,contours,hierarchy,CV_RETR_LIST,CV_CHAIN_APPROX_NONE, Point(0, 0));
      cvtColor(src_edge,dstIMAGE, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
      dstIMAGE=Scalar::all(0);
      HoughLinesP(src_edge,lines,1,CV_PI/180,80,length_min,10);
      Vec4i L=lines[0];
       line( dstIMAGE, Point(L[0], L[1]), Point(L[2], L[3]), Scalar(100,255,200), 1, CV_AA);
      float k0=((float)(L[3]-L[1]))/(L[2]-L[0]);
      for( size_t i = 1; i < lines.size(); i++ )
      {
        Vec4i l = lines[i];
        if((((float)(l[3]-l[1]))/(l[2]-l[0]))<-0.8||(((float)(l[3]-l[1]))/(l[2]-l[0]))>0.8){
        line(dstIMAGE, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 1, CV_AA);
        lines1.push_back(l);}
        else {lines2.push_back(l);
              line( dstIMAGE, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,150,150), 1, CV_AA);}
      }

      cout<<lines1.size()<<'\t'<<lines2.size()<<endl;
      Vec4i L1,L2;
      float k1,k2,b1,b2;
      for(i=0;i<lines1.size();i++){
            L1=lines1[i];
            for(j=0;j<lines2.size();j++){
                  L2=lines2[j];
                  b1=(L1[2]*L1[1]-L1[0]*L1[3])/(float)(L1[2]-L1[0]);
                  b2=(L2[2]*L2[1]-L2[0]*L2[3])/(float)(L2[2]-L2[0]);
                  k1=(float)(L1[3]-L1[1])/(L1[2]-L1[0]);
                  k2=(float)(L2[3]-L2[1])/(L2[2]-L2[0]);
                  x=(int)(b1-b2)/(k2-k1);
                  y=(int)(k1*x+b1);
                  if(x>=0&&x<dstIMAGE.cols&&y>=0&&y<=dstIMAGE.rows-1)
                  {      points.push_back(Point(x,y));
                        }
            }
      }
      //由点的密集程度来提取角点
      int density[points.size()],state_density;
      vector<int> label;
      int density_max=0,i_max=0;
      for(i=0;i<points.size();i++) density[i]=0;
      for(i=0;i<points.size();i++){
            for(j=0;j<points.size();j++){
                  if(getDistance(points[i],points[j])<10) density[i]++;
            }
      }
      if(points.size()>=4){
      while(label.size()<4) {
            for(i=0;i<points.size();i++)
            {
                  state_density=1;
                  for(j=0;j<label.size();j++)
                  if(i==label[j]||getDistance(points[label[j]],points[i])<length_min) state_density=0;
                  if(density[i]>density_max&&state_density)
                  {density_max=density[i];i_max=i;}
                  }
                  label.push_back(i_max);
                  cout<<i_max<<endl;
                  density_max=0;
      }
      cout<<getDistance(points[0],points[4])<<endl;
      for(i=0;i<label.size();i++)
      {
            cout<<label[i]<<endl;
            circle(dstIMAGE,points[label[i]],4,Scalar(180,255,100),1);
      }
      if(label.size()==4){
      //确定四个角点的位置
      Point2f center;
      for(i=0;i<4;i++){
            center.x+=points[label[i]].x;
            center.y+=points[label[i]].y;
      }
      center.x/=4;
      center.y/=4;
      vector<Point2f> top,buttom;
      Point2f tl,tr,bl,br;
      for(i=0;i<4;i++){
            if(center.y>points[label[i]].y) top.push_back(points[label[i]]);
            else buttom.push_back(points[label[i]]);
      }
      if(top[0].x<top[1].x) {tl=top[0];tr=top[1];}
      else {tl=top[1];tr=top[0];}
      if(buttom[0].x<buttom[1].x) {bl=buttom[0];br=buttom[1];}
      else {bl=buttom[1];br=buttom[0];}
      //构造变换矩阵
      vector<Point2f> quad_pts,corners;
      corners.push_back(Point2f(tl));
      corners.push_back(Point2f(tr));
      corners.push_back(Point2f(br));
      corners.push_back(Point2f(bl));

      Mat quad = Mat::zeros(src.size(),CV_8UC3);
      quad_pts.push_back(Point2f(0, 0));
      quad_pts.push_back(Point2f(quad.cols, 0));
      quad_pts.push_back(Point2f(quad.cols, quad.rows));
      quad_pts.push_back(Point2f(0, quad.rows));

      Mat transmtx = getPerspectiveTransform(corners, quad_pts);
      warpPerspective(src, quad, transmtx, quad.size());
      imshow("提取的纸张",quad);
      }
      else cout<<"无法识别纸张"<<endl;
   }
      else cout<<"无法识别纸张"<<endl;
      imshow("霍夫变换", dstIMAGE);
      waitKey(0);
}

int main()
{
      int way;
      cout<<"按1手动确定阈值，按2利用大津法自动确定阈值"<<endl;
      cout<<"按3局部自适应确定阈值，按4局部自适应确定阈值（优化）"<<endl;
      cout<<"按5三阈值分割,按6多阈值自适应分割"<<endl;
      cout<<"按7提取纸张（基于轮廓与角点），按8提取数字（利用局部自适应二值化)"<<endl;
      cin>>way;
      src=imread("提取数字2.jpg",1);
      imshow("原图",src);
      if(way<=6) namedWindow("二值化后的图像",WINDOW_AUTOSIZE);
      if(way==1){
      createTrackbar("阈值","二值化后的图像",&threshold_value,255,Thres_hold);
      Thres_hold(0,0);}
      else if(way==2) find_binary_value();
      else if(way==4||way==8) {createTrackbar("补偿值","二值化后的图像",&compensation,70,part_adapt_binary);
      createTrackbar("局部尺寸","二值化后的图像",&size_,100,part_adapt_binary);part_adapt_binary(0,0);}
      else if(way==3) part_adapt_binary_0();
      else if(way==5) multiple_binary();
      else if(way==6) multiple_binary_real();
      else if(way==7) {namedWindow("霍夫变换",WINDOW_AUTOSIZE);
                       createTrackbar("轮廓阈值","霍夫变换",&length_min,150,paper_segment);
                       paper_segment(0,0);
                      }
      waitKey(0);
      destroyWindow("二值化后的图像");
}
