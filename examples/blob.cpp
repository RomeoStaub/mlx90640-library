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

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#define MLX_I2C_ADDR 0x33
#define THRESHOLD 140
#define LOWBND 20
#define UPBND 35
#define IMAGE_SCALE 4
#define ROW 24
#define COL 32

using namespace std;
using namespace cv;


/// Global variables
float m = 255/(UPBND-LOWBND);
float b = -m*LOWBND;

Mat erosion_dst, dilation_dst;
int erosion_size = 2;
int dilation_size = 2;
int const max_elem = 2;
int const max_kernel_size = 10;
Rect bounding_rect;

bool showImage = 1;
bool showThresh = 1;
bool makeVideo = 0;

Mat img;
Mat imgThresh; 	//matrix storage for binary threshold image
Mat imgScaled;
Mat imgDiff;	
Mat im_color;
Mat mask;
Mat element;
Mat imgBack = Mat::zeros(Size(IMAGE_SCALE*ROW, IMAGE_SCALE*COL), CV_16UC1);;

// Global variables
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
int keyboard; //input from keyboard


// Valid frame rates are 1, 2, 4, 8, 16, 32 and 64
// The i2c baudrate is set to 1mhz to support these
#define FPS 16
#define FRAME_TIME_MICROS (1000000/FPS)

// Despite the framerate being ostensibly FPS hz
// The frame is often not ready in time
// This offset is added to the FRAME_TIME_MICROS
// to account for this.
#define OFFSET_MICROS 1200

	
// comparison function object
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
  /// Apply the dilation operation
  dilate(imgThresh, imgThresh, element );
}


int main(int argc, char* argv[])
{	
    auto frame_time = std::chrono::microseconds(FRAME_TIME_MICROS + OFFSET_MICROS);
    
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
    
    static float mlx90640To[768];
    pMOG2 = createBackgroundSubtractorMOG2(500,5,false); //MOG2 approach
    double learning_rate = 0.01;
    int frame_count = 0;
  

	VideoWriter out("out.avi",CV_FOURCC('M','J','P','G'), FPS, Size(IMAGE_SCALE*ROW, IMAGE_SCALE*COL));
	if(!out.isOpened()) {
		std::cout <<"Error! Unable to open video file for output." << std::endl;
	   std::exit(-1);
	}
	
	while((char)keyboard != 'q' && (char)keyboard != 27 ){
		frame_count += 1; 
		
		auto start = std::chrono::system_clock::now();
		
		// get Image
		MLX90640_GetFrameData(MLX_I2C_ADDR, frame); //takes a lot of time...
        MLX90640_InterpolateOutliers(frame, eeMLX90640);
        eTa = MLX90640_GetTa(frame, &mlx90640);
        MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);
		img = Mat(ROW, COL, CV_32FC1, &mlx90640To);
	
	
		flip(img, imgScaled,1); // flip image
		imgScaled = m*imgScaled+b;
		imgScaled.convertTo(imgScaled,CV_8UC1);
		// Apply bilinear interpolation
		resize(imgScaled, imgScaled, Size(IMAGE_SCALE*ROW, IMAGE_SCALE*COL), 0, 0, INTER_CUBIC);
		
		if(frame_count == 150) 
		{ 
			learning_rate =  0.0001;	//decrease learning rate after n frames
			cout << "Stopped training" << endl;
		}     
		pMOG2->apply(imgScaled, imgThresh,learning_rate);       

		Erosion(0,0);
		Dilation(0,0);
		//imgScaled.convertTo(imgScaled,CV_8UC1);
		applyColorMap(imgScaled, im_color, COLORMAP_JET); // 1630 us
		
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		// find contours
		findContours(imgThresh, contours, hierarchy,CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0)); //100 us

		// sort contours starting with biggest contour
		sort(contours.begin(), contours.end(), compareContourAreas);
		
				
		for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
		  {
			  if  (contourArea(contours[i])> 5*IMAGE_SCALE)
			  {
				//cout << "Countour area: " <<  contourArea(contours[i]) << endl;
				mask = Mat::zeros(imgScaled.size(), CV_8UC1);
				drawContours(mask,contours,i,Scalar(255),FILLED,8,hierarchy);           
				Mat masked(imgScaled.size(),CV_8UC1,Scalar(0));
				imgScaled.copyTo(masked,mask);
				Moments m = moments(masked,false);
				Point2d p(m.m10/m.m00, m.m01/m.m00);
				drawMarker(im_color,p,Scalar(0,0,0),0,1*IMAGE_SCALE,1*IMAGE_SCALE,8);
				putText(im_color, to_string(i+1), p, FONT_HERSHEY_PLAIN,0.5*IMAGE_SCALE,  Scalar(0,0,0));
			  }
		   }
		   

		 
		if (makeVideo)
		{  
			// Save frame to video
			out << im_color;
		}
		
		// show the resultant threshold image
		if (showThresh)
		{
			createTrackbar( "Erosion Kernel size:\n 2n +1", "Morph",&erosion_size, max_kernel_size,Erosion);
			createTrackbar( "Dilation Kernel size:\n 2n +1", "Morph",&dilation_size, max_kernel_size,Dilation );
			namedWindow( "Morph",  WINDOW_NORMAL);
			imshow("Morph", imgThresh);
	    }
	    
	    
	    // show the resultant image
        if(showImage)
        {
			// show the resultant image
			namedWindow( "Contours", WINDOW_NORMAL);
			imshow("Contours", im_color);
			keyboard = waitKey(1); // NEEDED TO SHOW IMAGE
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
        this_thread::sleep_for(std::chrono::microseconds(frame_time - elapsed));
        
	}	
	//destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
    return 0;
}




