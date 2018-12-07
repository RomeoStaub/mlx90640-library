#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include "headers/MLX90640_API.h"
#include "opencv2/core.hpp"
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <numeric>
#include <sstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <algorithm>
#include <math.h>
#include <wiringPi.h>
#include "headers/MLX90640_CONSTANTS.h"


using namespace std;
using namespace cv; 
///  Global variables
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
	
int keyboard; //input from keyboard

int sigma = 4;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

//temperature mapping
float m = 255/(UPBND-LOWBND);
float b = -m*LOWBND;


//Servo mapping
float mSV = (SVHIGH-SVLOW)/180.0;
float bSV = SVLOW;

//MLX90640 
static uint16_t eeMLX90640[832];
float emissivity = 1;
uint16_t frame[834];
float eTa;
int subpage;
static float mlx90640To[768];

paramsMLX90640 mlx90640;
int mode;
int refresh;

// Morph operation
int morph_elem = 0;
int morph_size = 0;
int const max_operator = 4;

bool makeImage = 1; 
bool showImage = 1; // for debugging
bool showThresh = 1; 
bool showMorph = 1;
bool makeVideo = 0;
bool saveImg = 0;
bool loadImg = 1;
bool debugMode = 0;
bool createYaml = 0;
bool loadYaml = 0;
bool saveStdDev = 0;
bool saveTrajectory = 1;
bool runServo = 0;

Mat img 		= Mat::zeros(Size(COL, ROW), CV_32F);
Mat imgScaled 	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgBack		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat acc 		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat sqacc 		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgLevel 	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgThresh	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_8UC1); 	//matrix storage for binary threshold image
Mat imgMorph 	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_8UC1);
Mat imgColor;
Mat mask;
Mat element;
Mat stdDev;

Mat imgPnt(3,1,CV_32FC1);
Mat worldPnt(3,1,CV_32FC1);

vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
	
/**  @function compareContourAreas  */
bool compareContourAreas (vector<cv::Point> contour1,vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(Mat(contour1)) );
    double j = fabs( contourArea(Mat(contour2)) );
    return (i > j);
}
  
void ClearScreen()
    {
    cout << string( 100, '\n' );
    }

void setAngle(int angle)
{
	// cout << mSV*angle+bSV << endl;
	pwmWrite(18,mSV*angle+bSV);
	//delay(10);
}

int getImg(Mat &img) {
	int status;
    status = MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
    if (status == -8 )
    {
		cout << "frame not ready" << endl;
	}
	
	//subpage = MLX90640_GetSubPageNumber (frame);
	//cout << "subPage: " << subpage << endl;		
	eTa = MLX90640_GetTa(frame, &mlx90640);
	MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);
	MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels, mlx90640To, mode, &mlx90640);
	MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels, mlx90640To, mode, &mlx90640);
	img = Mat(ROW, COL, CV_32FC1, &mlx90640To);
	//cout << "Img: " << endl << img << endl;
    return status;
}


int configureMLX90640()
{
	int status;
	
	status = MLX90640_SetDeviceMode(MLX_I2C_ADDR, 1);
    status = MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
            
    switch(FPS){
        case 1:
            status = MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b001);
            break;
        case 2:
            status = MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
            break;
        case 4:
            status = MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b011);
            break;
        case 8:
            status = MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b100);
            break;
        case 16:
            status = MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b101);
            break;
        case 32:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b110);
            break;    
        default:
            printf("Unsupported framerate: %d", FPS);
            return -1;
    }
    
    status = MLX90640_SetChessMode(MLX_I2C_ADDR);
    printf("Configured...\n");
    
    status = MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
    status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
    mode = MLX90640_GetCurMode(MLX_I2C_ADDR);
    refresh = MLX90640_GetRefreshRate(MLX_I2C_ADDR);  // curRR = 0x05(16Hz) as this is the actual
    //cout << "refresh " << refresh << endl;
    
    printf("EE Dumped...\n");
    return status;
    
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void writeCSV(string filename, Mat m)
{
   ofstream myfile;
   myfile.open(filename.c_str());
   myfile<< cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
   myfile.close();
}
    
float convert2D(float arr[], int i, int j, int n)//n defines number of columns
{
    return (arr[(i*n + j)]);
}


float BilinearInterpolation(float q11, float q12, float q21, float q22, int x1, int x2, int y1, int y2, float x, float y) //from https://helloacm.com/cc-function-to-compute-the-bilinear-interpolation/
{
    float x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}


Point2d getPos(float x, float y)
{
	
	Point2d pC;
	int x1,x2,y1,y2;
	float q11,q12,q21,q22;
	
	x = x/IMAGE_SCALE;
	y = y/IMAGE_SCALE;
	
	x1 = static_cast <int> (floor(x));
	y1 = static_cast <int> (floor(y));
	x2 = x1+1;
	y2 = y1+1;
	//get x Position
	q11 = posX[(y1*32 + x1)];
	q12 = posX[(y2*32 + x1)];
	q21 = posX[(y1*32 + x2)];
	q22 = posX[(y2*32 + x2)];
	pC.x = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
	//get y Position
	q11 = posY[(y1*32 + x1)];
	q12 = posY[(y2*32 + x1)];
	q21 = posY[(y1*32 + x2)];
	q22 = posY[(y2*32 + x2)];
	pC.y = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
	return pC;
	
}

void makePos(float Z)
{
	for (int i = 0;i<768;i++)
	{	//negative value to adjust coordinate system according to image coord
		posX[i] = -Z*tan(angleX[i] * PI / 180.0 );  
		posY[i] = -Z*tan(angleY[i] * PI / 180.0 );    
	}
	return;
}


int main(int argc, char* argv[])
{
	//Create look up table for given height (H-h)
	makePos(H-h);
	auto frame_time = std::chrono::microseconds(FRAME_TIME_MICROS + OFFSET_MICROS);
	
	double u0 = (float) COL*IMAGE_SCALE/2.;
	double v0 = (float) ROW*IMAGE_SCALE/2.;
	
	
	double Lx = 2*(H-h)*tan(FOVX/2.0*PI/180.0);
	double Ly = 2*(H-h)*tan(FOVY/2.0*PI/180.0);

	//cout << "Lx " << Lx << endl;
	//cout << "Ly " << Ly << endl;
	//cout << "u0 " << u0 << endl;
	//cout << "v0 " << v0 << endl;
	float f = (float) u0/(tan(FOVX/2* PI / 180.0 )); // [pixel] specified FoV is calculated for the wider direction, in this case for the 32 pixels
	//cout << "f " << f << endl;
	float fovy = (float) 2*atan2(v0,f)*180/PI;
	//cout << "fovy " << fovy << endl;
	float fy = (float) v0/(tan(FOVY/2.* PI / 180.0 ));
	//cout << "fy " << fy << endl;
	
	//cout << "output" << endl << 2*u0*(H-h)/f << endl;
	Mat K(3,3,CV_32FC1);
    setIdentity(K);
    K.at<float>(0,0) = f;
    K.at<float>(1,1) = f;
    K.at<float>(0,2) = u0;
    K.at<float>(1,2) = v0;
	
    cout << "K " << endl << K << endl;
    
    Mat R_C_W(3,3,CV_32FC1);
    setIdentity(R_C_W);
    R_C_W.at<float>(0,0) = -1;
    R_C_W.at<float>(2,2) = -1;
    cout << "Rotation Matrix " << endl << R_C_W << endl;
    
    Mat t_C_W(3,1,CV_32FC1);
    t_C_W.at<float>(0) = 0;
    t_C_W.at<float>(1) = -YDIST;
    t_C_W.at<float>(2) = H;
    
	cout << "translationVector " << endl << t_C_W << endl;
	
	string filename = "cameraMatrix.yaml";
    
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "K" << K;
	fs << "R_C_W" << R_C_W;
	fs << "t_C_W" << t_C_W;
	fs.release();                                       // explicit close
	cout << "Write Done." << endl;
     
	
	
   if (wiringPiSetupGpio() < 0) return 1;  // Initializes wiringPi using the Broadcom GPIO pin numbers
   pinMode(18, PWM_OUTPUT);
   pwmSetMode (PWM_MODE_MS);
   pwmSetClock(384); //clock at 50kHz (20us tick)
   pwmSetRange(1000); //range at 1000 ticks (20ms)
	    
    printf("Starting...\n");
    int success = 1;
    while(success != 0)
	{
		success = configureMLX90640();
		cout << "config success: " << success << endl;
    }
    printf("configureMLX90640 successful\n");

    pMOG2 = createBackgroundSubtractorMOG2(500,sigma*sigma,false); //MOG2 approach
    double learning_rate = -1;
    int frame_count = 0;

	VideoWriter out; // create video output
	
	if (makeVideo)
	{
		out.open("out.avi", CV_FOURCC('M','J','P','G'), FPS/2, Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), true);	// set video output to 15 fps and MJPG
		//Create and initialize the VideoWriter object 
		if(!out.isOpened()) {
			cout <<"Error! Unable to open video file for output." << std::endl;
		    exit(-1);
		}
	}

	if (saveImg) // empty folder
	{
		system("exec rm -r images/*");
	}
	
	if (loadImg)
	{
		String folder = "images/"; // again we are using the Opencv's embedded "String" class
		glob(folder, filenames); // new function that does the job ;-)
	}
	
	ofstream trajectory;
	if (saveTrajectory)
	{
	   trajectory.open("trajectory.csv");
	}
		
	while((char)keyboard != 'q' && (char)keyboard != 27 ){
		
		auto start = std::chrono::system_clock::now();
		auto startTime = std::chrono::system_clock::now();

		frame_count += 1; 
		if (loadImg)
		{
			img = imread(filenames[frame_count],CV_LOAD_IMAGE_UNCHANGED);
			if(!img.data)
			{
				cerr << "Finished with frame " << frame_count << endl;
				break;
			}
		}
		else
		{
			success = 1;
			while (success != 0)
			{
				success = getImg(img);
				//cout << "success: " << success << endl;
			}			
		}

		if (saveImg)
		{
			char imgName[100];
			sprintf(imgName, "images/img_%06d.exr", frame_count);
			imwrite(imgName, img);
		}
		
		auto endTime = std::chrono::system_clock::now();
		flip(img, imgScaled,1);
		// Apply cubic interpolation
		resize(imgScaled, imgScaled, Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), 0, 0, INTER_CUBIC);
						
		switch(BACKGROUNDMETHOD){
        case 1:
        	//imgScaled.convertTo(imgScaled,CV_32F);
        	if (frame_count == 1)
			{
				cout << "Calibrating Background ..." << endl;
			}  
			
			
			if (frame_count < BSIGNORE)
			{
				continue;
			}
			else if(frame_count < BSSAMPLE) 
			{
				accumulate(imgScaled, acc);
				accumulateSquare(imgScaled, sqacc);
				add(imgScaled,imgBack,imgBack);
				continue;
			}
			else if (frame_count == BSSAMPLE)
			{
				imgBack = acc/(BSSAMPLE-BSIGNORE);
				// cout << "Background M" << imgBack << endl;
				Mat sqaccM = sqacc/(BSSAMPLE-BSIGNORE);
				Mat imgBack2 = imgBack.mul(imgBack);
				Mat sig2 = sqaccM-imgBack2;
				sqrt(sig2,stdDev);
				// cout << "Mean " << endl << imgBack << endl;
				//Scalar mean, stddev;
				//meanStdDev(imgBack, mean, stddev);
				//cout << "Mean: " << mean[0] << "   StdDev: " << stddev[0] << endl;
				// cout << "stdDev: " << endl << stdDev << endl;	
				// meanStdDev(stdDev, mean, stddev);
				//cout << "Mean stDev " << mean[0] << "   stDev StdDev: " << stddev[0] << endl;
				//cout << "Background" << imgBack << endl;
				cout << "Calibration done" << endl;
				getchar();
				if (saveStdDev)
				{
					writeCSV("stdDev.csv", stdDev);
					/*
					for(int i=0; i<stdDev.rows; i++)
					{
						for(int j=0; j<stdDev.cols; j++)
						{
							stdDevData << stdDev.at<float>(i, j);
						}
						stdDevData << endl;
					}
					*/
				}
				
				continue;
			}
			else
			{
				imgLevel = imgScaled-imgBack-sigma*stdDev;			
				//cout << "After background substraction " << imgLevel << endl;
				threshold(imgLevel,imgThresh,0,1,0);
				//cout << "ImgThresh Depth " << imgThresh.type() << ",Rows " << imgThresh.cols << "Cols " << imgThresh.rows <<endl;
				//cout << "imgThresh " << imgThresh << endl;
				imgThresh.convertTo(imgThresh,CV_8UC1);
				//cout << "ImgThresh Depth " << imgThresh.type() << ",Rows " << imgThresh.cols << "Cols " << imgThresh.rows <<endl;
				//cout << "imgThresh " << imgThresh << endl;
				//imgScaled.convertTo(imgScaled,CV_8UC1);
				break;
			}
			
        case 2:
			pMOG2->apply(m*imgScaled+b,imgThresh,learning_rate);
			if (frame_count == 1)
			{
				cout << "Calibrating Background ..." << endl;
				continue;
			}   
			else if(frame_count < BSSAMPLE) 
			{
				continue;
			}
			else if(frame_count == BSSAMPLE) 
			{ 
				//learning_rate =  0.001/FPS;	//decreasing learning rate
				learning_rate =  0;	//decreasing learning rate
				cout << "Calibration done" << endl;
			}
			break;
        default:
            printf("Unsupported backgroundsubstraction method: %d", BACKGROUNDMETHOD);
            return 1;
		}

		 Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
		 /// Apply the specified morphology operation
		 morphologyEx( imgThresh, imgMorph, 0 , element );
		 /// Apply the specified morphology operation
		 morphologyEx( imgMorph, imgMorph, 1 , element );
		 /// Default start
		 //Morphology_Operations( 0, 0 );
		
		//Erosion(0,0);
		//Dilation(0,0);
		
		
		// find contours
		//findContours(imgMorph, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //100 us
		findContours(imgMorph, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		// sort contours starting with biggest contour
		sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Point2f>pointList;
		vector<Point2f>pointListDist;

		
		for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
		  {
			  if  (contourArea(contours[i])> MINAREA*IMAGE_SCALE*IMAGE_SCALE)
			  {

				Point2d pC;
				int N = contours[i].size();
				for( int j = 0; j< N; j++) // iterate through each pixel pair inside contour. 
				{
					pC += getPos(contours[i][j].x,contours[i][j].y);
				}
				pC = pC/N;
				pointListDist.push_back(pC);

				//Second Approach
				mask = Mat::zeros(imgScaled.size(), CV_8UC1);
				drawContours(mask,contours,i,Scalar(255),FILLED,8,hierarchy);  
				Mat masked(imgScaled.size(),CV_8UC1,Scalar(0));
				imgScaled.copyTo(masked,mask);
				Moments m = moments(masked,false);
				Point2d p(m.m10/m.m00, m.m01/m.m00); //u0, v0
				pointList.push_back(p);
		    }
		 }
		
		if (makeImage)
			{
			// ClearScreen();
			imgScaled = m*imgScaled+b;
			imgScaled.convertTo(imgScaled,CV_8UC1);
			applyColorMap(imgScaled, imgColor, COLORMAP_JET); // 1630 us  

			for( int i = 0; i< pointList.size(); i++ ) // iterate through each contour.
			{
				imgColor.convertTo(imgColor,CV_8UC1);
				drawMarker(imgColor,pointList[i],Scalar(0,0,0),0,1*IMAGE_SCALE,1*IMAGE_SCALE,8);
				putText(imgColor, to_string(i+1), pointList[i], FONT_HERSHEY_PLAIN,0.5*IMAGE_SCALE,  Scalar(0,0,0));
			}
		}
		
		if (makeVideo)	out << imgColor; // add frame to video
		
	    // show the resultant image
        if(showImage)
        {	
			if(showMorph)
			{
				imgMorph = 255*imgMorph;
				imgMorph.convertTo(imgMorph,CV_8UC1);
				cvtColor(imgMorph, imgMorph, COLOR_GRAY2BGR);
				//cout << "ImgMorph Depth" << imgMorph.type() << "Rows" << imgMorph.cols << "Cols" << imgMorph.rows <<endl;
				//cout << "imgColor Depth" << imgColor.type() << "Rows" << imgColor.cols << "Cols" << imgColor.rows <<endl;
				hconcat(imgMorph,imgColor,imgColor);

			}
			
		    if(showThresh)
		    {
				//createTrackbar( "Erosion Kernel size:\n 2n +1", "Contours",&erosion_size, max_kernel_size,Erosion);
			    //createTrackbar( "Dilation Kernel size:\n 2n +1", "Contours",&dilation_size, max_kernel_size,Dilation );
			    //string ty =  type2str( M.type() );
				//printf("Matrix: %s %dx%d \n", ty.c_str(), M.cols, M.rows );

			    //cout << "imgThresh Depth" << imgThresh.type() << "Rows" << imgThresh.cols << "Cols" << imgThresh.rows <<endl;
				//cout << "imgColor Depth" << imgColor.type() << "Rows" << imgColor.cols << "Cols" << imgColor.rows <<endl;
				
				imgThresh = 255*imgThresh;
				cvtColor(imgThresh, imgThresh, COLOR_GRAY2BGR);
				hconcat(imgThresh,imgColor,imgColor);
			}
						
			// show the resultant image
			namedWindow( "Contours", WINDOW_NORMAL);
			imshow("Contours", imgColor);
		}

		
		
		for( int i = 0; i< pointList.size(); i++ ) // iterate through each contour. 
		{
			imgPnt.at<float>(0) = pointList[i].x; //u
			imgPnt.at<float>(1) = pointList[i].y; //v
			imgPnt.at<float>(2) = 1;
			cout << "Pixel coordinates: " << imgPnt << endl;
			
			
			Mat leftSideMat = R_C_W.inv()*K.inv()*imgPnt;
			Mat rightSideMat = R_C_W.inv()*t_C_W;
			double lambda = (h + rightSideMat.at<float>(2,0))/leftSideMat.at<float>(2,0);
			worldPnt = R_C_W.inv()*(lambda*K.inv()*imgPnt-t_C_W);
			cout << i+1 << ".Human in  World coordinates" << endl << worldPnt << endl;
			//cout << "with area of " << contourArea(contours[i]) << "pixel²" << endl;
			// show Angle of first person only
			
			
				float theta = atan2(worldPnt.at<float>(0),worldPnt.at<float>(1))*180/PI; //atan2(y,x)
				cout << "Theta" << theta << endl;
				cout << "Theta servo"  << 90-theta <<endl;
				if (runServo) setAngle(90-theta);
				
				if (saveTrajectory)
				{	
					trajectory << frame_count << ",";
					trajectory << imgPnt.at<float>(0) << ",";
					trajectory << imgPnt.at<float>(1) << ",";
					trajectory << worldPnt.at<float>(0) << ",";
					trajectory << worldPnt.at<float>(1) << ",";
					trajectory << worldPnt.at<float>(2) << ",";
					trajectory << theta <<  ",";			
					trajectory << pointListDist[i].x << ",";
					trajectory << pointListDist[i].y << ",";
					trajectory << endl;
					

				}
			
			//Map pixel coordinates to image coordinates
			/*
			pointList[i].x= (pointList[i].x-u0) / px*Lx; // ?
			pointList[i].y= (pointList[i].y-v0) / py*Ly; ?
			cout << "     " << i+1 << " Human detected @: " << pointList[i] << endl;
			*/
		}
		
		

		

		
		
		if(debugMode)
		{
			keyboard = waitKey(50); // Stop program by pressing escape or q 
			if ((char)keyboard == 'b')
			{
				frame_count -= 2; 
				waitKey(0);
			}
			else
			{
				waitKey(0);
			}
			
		}
		else
		{
			waitKey(1);
		}
			
		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
		//cout << "Measured time " << elapsedTime.count() << endl;
		//cout << "Percentage of calulated time " << 100.0/elapsed.count()*elapsedTime.count()<< endl;
		
		if (frame_time.count()-elapsed.count() < 0)
		{
			/*
			cout << frame_count << "th Frame not ready" << endl;
			cout << "Take smaller FPS" << endl;
			cout << "Frame time" << frame_time.count()<< " us" << endl;    
			cout << "Elaped time" << elapsed.count()<< " us" << endl;    
			cout << "Wait time" << frame_time.count()-elapsed.count()<< " us" << endl; 
			*/
		}
		//this_thread::sleep_for(chrono::microseconds(frame_time - elapsed));

	}
	if (saveTrajectory)	trajectory.close();
	//destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
    return 0;
}




