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
#include "opencv2/video/background_segm.hpp"
#include "opencv2/bgsegm.hpp"


	
#define MLX_I2C_ADDR 0x33
#define THRESHOLD 140
#define LOWBND 22
#define UPBND 35
#define IMAGE_SCALE 2.0
#define ROW 24.0
#define COL 32.0
#define H 2450 //ceiling height in mm
#define h 1500  //subject height in mm
#define FOVX 110.0
#define FOVY 75.0
#define PI 3.14159265
#define YDIST 1500
/// Servo Parameters
#define SVLOW 10
#define SVHIGH 110
float mSV = 0.9/1.8;
float bSV = 20;

// Valid frame rates are 1, 2, 4, 8, 16, 32 and 64
// The i2c baudrate is set to 1mhz to support these
#define FPS 16
#define FRAME_TIME_MICROS (1000000/FPS)

// Despite the framerate being ostensibly FPS hz
// The frame is often not ready in time
// This offset is added to the FRAME_TIME_MICROS
// to account for this.
//#define OFFSET_MICROS 1200
#define OFFSET_MICROS 1600

// Choose Backgroundsubstraction method
// 1 =  Substract average of first 30
// 2 =  MOG2 Background subtractor
#define BACKGROUNDMETHOD 2


using namespace std;
using namespace cv; 
///  Global variables
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
Ptr<BackgroundSubtractor> pMOG; //MOG Background subtractor
	
int keyboard; //input from keyboard

int sigma = 4;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

float m = 255/(UPBND-LOWBND);
float b = -m*LOWBND;

Mat erosion_dst, dilation_dst;
int erosion_size = IMAGE_SCALE;
int dilation_size = IMAGE_SCALE;
int const max_elem = 2;
int const max_kernel_size = 10;
Rect bounding_rect;

bool showImage = 1;
bool showThresh = 1; // for debugging
bool makeVideo = 0;
bool saveImg = 0;
bool loadImg = 1;
bool debugMode = 0;	

Mat img;
Mat imgThresh; 	//matrix storage for binary threshold image
Mat imgScaled;
Mat imgColor;
Mat mask;
Mat element;
Mat imgBack = Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_16UC1);;

vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class

	
/**  @function compareContourAreas  */
bool compareContourAreas (vector<cv::Point> contour1,vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(Mat(contour1)) );
    double j = fabs( contourArea(Mat(contour2)) );
    return (i > j);
}

/**  @function Erosion  */
void Erosion( int, void* )
{
  Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point(erosion_size, erosion_size ) );
  /// Apply the erosion operation
  erode(imgThresh, imgThresh, element );
}

/** @function Dilation */
void Dilation( int, void* )
{
  Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  // Apply the dilation operation
  dilate(imgThresh, imgThresh, element );
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

int main(int argc, char* argv[])
{	
	
	auto frame_time = std::chrono::microseconds(FRAME_TIME_MICROS + OFFSET_MICROS);
	
	double u0 = (float) COL*IMAGE_SCALE/2.;
	double v0 = (float) ROW*IMAGE_SCALE/2.;
	
	
	double Lx = 2*(H-h)*tan(FOVX/2.0*PI/180.0);
	double Ly = 2*(H-h)*tan(FOVY/2.0*PI/180.0);

	//cout << "Lx " << Lx << endl;
	//cout << "Ly " << Ly << endl;
	// cout << "u0 " << u0 << endl;
	// cout << "v0 " << v0 << endl;
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
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = u0;
    K.at<float>(1,2) = v0;
	
    //cout << "K " << endl << K << endl;
    
    Mat R_C_W(3,3,CV_32FC1);
    setIdentity(R_C_W);
    R_C_W.at<float>(0,0) = -1;
    R_C_W.at<float>(2,2) = -1;
    //cout << "Rotation Matrix " << endl << R_C_W << endl;
    
    Mat t_C_W(3,1,CV_32FC1);
    t_C_W.at<float>(0) = 0;
    t_C_W.at<float>(1) = -YDIST;
    t_C_W.at<float>(2) = H;
    
	//cout << "translationVector " << endl << t_C_W << endl;


			
   if (wiringPiSetupGpio() < 0) return 1;  // Initializes wiringPi using the Broadcom GPIO pin numbers
   pinMode(18, PWM_OUTPUT);
   pwmSetMode (PWM_MODE_MS);
   pwmSetClock(384); //clock at 50kHz (20us tick)
   pwmSetRange(1000); //range at 1000 ticks (20ms)

	    
    printf("Starting...\n");
    static uint16_t eeMLX90640[832];
    float emissivity = 1;
    uint16_t frame[834];
    float eTa;
    
    MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
            
    switch(FPS){
        case 1:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b001);
            break;
        case 2:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
            break;
        case 4:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b011);
            break;
        case 8:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b100);
            break;
        case 16:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b101);
            break;
        default:
            printf("Unsupported framerate: %d", FPS);
            return 1;
    }
    
    MLX90640_SetChessMode(MLX_I2C_ADDR);
    printf("Configured...\n");
    paramsMLX90640 mlx90640;
    MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
    MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
    int refresh = MLX90640_GetRefreshRate(MLX_I2C_ADDR);
    printf("EE Dumped...\n");
    
    int frames = 30;
    int subpage;
    static float mlx90640To[768];
    //pMOG2 = createBackgroundSubtractorMOG2(1000,sigma*sigma,false); //MOG2 approach
    pMOG2 = createBackgroundSubtractorGMG(1000,500,false); //MOG2 approach
    pMOG = createBackgroundSubtractorGMG(1000,500,false); //MOG2 approach



    double learning_rate = -1;
    int frame_count = 0;
    
	VideoWriter out; // create video output
	
	if (makeVideo)
	{
		out.open("out.avi", CV_FOURCC('M','J','P','G'), FPS, Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), true);	// set video output to 15 fps and MJPG
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
		
	while((char)keyboard != 'q' && (char)keyboard != 27 ){
		auto start = std::chrono::system_clock::now();
		
		frame_count += 1; 
		if (loadImg)
		{
			img = imread(filenames[frame_count],-1);
			if(!img.data)
			{
				cerr << "Finished with frame " << frame_count << endl;
				break;
			}
		}
		else
		{
			MLX90640_GetFrameData(MLX_I2C_ADDR, frame); //takes a lot of time...
			MLX90640_InterpolateOutliers(frame, eeMLX90640);
			eTa = MLX90640_GetTa(frame, &mlx90640);
			// subpage = MLX90640_GetSubPageNumber(frame);
			MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);
			img = Mat(ROW, COL, CV_32FC1, &mlx90640To);
		}
		
		if (saveImg)
		{
			char imgName[100];
			sprintf(imgName, "images/img_%06d.exr", frame_count);
			imwrite(imgName, img);
		}

		flip(img, imgScaled,1);
		imgScaled = m*imgScaled+b;
		imgScaled.convertTo(imgScaled,CV_8UC1);
		// Apply cubic interpolation
		resize(imgScaled, imgScaled, Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), 0, 0, INTER_CUBIC);
		switch(BACKGROUNDMETHOD){
        case 1:
			if(frame_count < 150) 
			{
				cout << "Calibrating Background ..." << endl;
				imgScaled.convertTo(imgScaled,CV_16UC1);
				add(imgScaled,imgBack,imgBack);
				cout << "asdf"<< endl;
				continue;
			}
			else if (frame_count == 150)
			{
				
				imgBack = imgBack / 150.0;
				imgBack.convertTo(imgBack,CV_8UC1);
				cout << "Calibration done" << endl;
				continue;

			}
			else
			{
				imgScaled = imgScaled - imgBack; //substract Background
				threshold(imgScaled,imgThresh,15,255,0);
				imgThresh.convertTo(imgThresh,CV_8UC1);
				break;
			}
        case 2:
			pMOG2->apply(imgScaled, imgThresh,learning_rate);
			if (frame_count == 1)
			{
				cout << "Calibrating Background ..." << endl;
				continue;
			}   
			else if(frame_count < 150) 
			{
				continue;
			}
			else if(frame_count == 150) 
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
	
		Erosion(0,0);
		Dilation(0,0);
		applyColorMap(imgScaled, imgColor, COLORMAP_JET); // 1630 us
		
		// find contours
		findContours(imgThresh, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //100 us

		// sort contours starting with biggest contour
		sort(contours.begin(), contours.end(), compareContourAreas);
		
		vector<Point2f>pointList;
		
		for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
		  {
			  if  (contourArea(contours[i])> 5*IMAGE_SCALE*IMAGE_SCALE)
			  {
				//cout << "Countour area: " <<  contourArea(contours[i]) << endl;
				mask = Mat::zeros(imgScaled.size(), CV_8UC1);
					drawContours(mask,contours,i,Scalar(255),FILLED,8,hierarchy);           
				Mat masked(imgScaled.size(),CV_8UC1,Scalar(0));
				imgScaled.copyTo(masked,mask);
				Moments m = moments(masked,false);
				Point2d p(m.m10/m.m00, m.m01/m.m00); //u0, v0
				pointList.push_back(p);
		    }
		 }
		
		
		// ClearScreen();

			    
	    for( int i = 0; i< pointList.size(); i++ ) // iterate through each contour. 
		{
				drawMarker(imgColor,pointList[i],Scalar(0,0,0),0,1*IMAGE_SCALE,1*IMAGE_SCALE,8);
				putText(imgColor, to_string(i+1), pointList[i], FONT_HERSHEY_PLAIN,0.5*IMAGE_SCALE,  Scalar(0,0,0));
		}
		
		if (makeVideo)	out << imgColor; // add frame to video
		
	    // show the resultant image
        if(showImage)
        {	
		    if(showThresh)
		    {
				createTrackbar( "Erosion Kernel size:\n 2n +1", "Contours",&erosion_size, max_kernel_size,Erosion);
			    createTrackbar( "Dilation Kernel size:\n 2n +1", "Contours",&dilation_size, max_kernel_size,Dilation );
				cvtColor(imgThresh, imgThresh, COLOR_GRAY2BGR);
				hconcat(imgColor,imgThresh,imgColor);
			}
			// show the resultant image
			namedWindow( "Contours", WINDOW_NORMAL);
			imshow("Contours", imgColor);
		}
		
		
		//cout << "Frame: " << frame_count << endl;
		for( int i = 0; i< pointList.size(); i++ ) // iterate through each contour. 
		{
			Mat uvPoint(3,1,CV_32FC1);
			uvPoint.at<float>(0) = pointList[i].x; //u
			uvPoint.at<float>(1) = pointList[i].y; //v
			uvPoint.at<float>(2) = 1;
			Mat leftSideMat = R_C_W.inv()*K.inv()*uvPoint;
			Mat rightSideMat = R_C_W.inv()*t_C_W;
			double lambda = (h + rightSideMat.at<float>(2,0))/leftSideMat.at<float>(2,0);
			Mat P = R_C_W.inv()*(lambda*K.inv()*uvPoint-t_C_W);
			cout << i+1 << ". Human at" << endl << P << endl;
			//cout << "with area of " << contourArea(contours[i]) << "pixel²" << endl;
			
			// show Angle of first person only
			
			if (i == 0)
			{
				float theta = atan2(P.at<float>(0),P.at<float>(1))*180/PI; //atan2(y,x)
				cout << "Theta" << theta << endl;
				cout << "Theta servo"  << 90-theta <<endl;
				setAngle(90-theta);
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
		
				
		//auto startTime = std::chrono::system_clock::now();
		//auto endTime = std::chrono::system_clock::now();
		auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        //auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        //cout << "Measured time " << elapsedTime.count() << endl;
        //cout << "Percentage of calulated time " << 100.0/elapsed.count()*elapsedTime.count()<< endl;
        
        if (frame_time.count()-elapsed.count() < 0)
        {
			
			cout << frame_count << "th Frame not ready" << endl;
			/*
			cout << "Take smaller FPS" << endl;
			cout << "Frame time" << frame_time.count()<< " us" << endl;    
			cout << "Elaped time" << elapsed.count()<< " us" << endl;    
			cout << "Wait time" << frame_time.count()-elapsed.count()<< " us" << endl;
			*/ 
	    }
        this_thread::sleep_for(chrono::microseconds(frame_time - elapsed));
        
	}	
	//destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
    return 0;
}




