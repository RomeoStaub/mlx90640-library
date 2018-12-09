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
#include "lib/background_substraction.h"
#include "lib/human_detection.h"


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
int morph_elem = 0; // 0: Rect - 1: Cross - 2: Ellipse
int morph_size = IMAGE_SCALE;
int const max_operator = 4;

bool makeImage = 1; 
bool showImage = 1; // for debugging
bool showThresh = 0; 
bool showMorph = 0;
bool makeVideo = 0;
bool saveImg = 0;	
bool loadImg = 0;
bool debugMode = 0;
bool createYaml = 0;
bool loadYaml = 0;
bool saveStdDev = 0;
bool saveTrajectory = 1;
bool runServo = 0;
bool saveMat = 0;

Mat img 		= Mat::zeros(Size(COL, ROW), CV_32F);
Mat imgFlip		= Mat::zeros(Size(COL, ROW), CV_32F);
Mat imgInt 		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgSmooth 	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgBack		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat acc 		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat sqacc 		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgLevel 	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgThresh	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F); 	//matrix storage for binary threshold image
Mat imgMorph 	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_8UC1);
Mat imgMask 	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_8UC1);

Mat imgColor;
Mat mask;
Mat element;
Mat imgStd;
Mat imgRaw= Mat::zeros(Size(COL, ROW), CV_32F);;
Mat imgRaw2= Mat::zeros(Size(COL, ROW), CV_32F);;
Mat imgRawStd= Mat::zeros(Size(COL, ROW), CV_32F);;
Mat imgDebug;


double minVal;
double maxVal;
Point minLoc;
Point maxLoc;

int gaussSigma = IMAGE_SCALE;
// make odd filter size that contains 99% of data
int filterSize = gaussSigma*3+gaussSigma % 2-1;


// Variables for running gaussian average
//float rho = 1/(BSSAMPLE-BSIGNORE);
float rho = 0.05;
double kThresh = 3; // Z-Score

Mat imgMean		= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgZScore	= Mat::zeros(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgVar		= Mat::ones(Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), CV_32F);
Mat imgSub;
Mat imgDist;
Scalar tempVal;
float imgStdMean;

Mat camPnt(3,1,CV_32FC1);
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


//Gets image from MLX90640
int getImg(Mat &img) {
	int status = 1;
	/*
	while (status != 0)
	{
		status = MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
	}
	*/
	status = MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
	eTa = MLX90640_GetTa(frame, &mlx90640);
	MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To);
	MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels, mlx90640To, mode, &mlx90640);
	MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels, mlx90640To, mode, &mlx90640);
	img = Mat(ROW, COL, CV_32FC1, &mlx90640To);

	
	if (!checkRange(img, true))
	{
		//cout << "mlx contains NaN" << endl <<  &mlx90640To << endl;
		status = 1;
	}
    return status;
}


int configureMLX90640()
{
	int status;
	status = MLX90640_SetDeviceMode(MLX_I2C_ADDR, 1);
    status = MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    status = MLX90640_SetResolution(MLX_I2C_ADDR,0x03);
    
    // 
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
	// Any values out of acceptable range
	double epsilion = 0.0001;
	if (x > 31-epsilion)
	{
		x = 31-epsilion;
	}
	if (y > 23-epsilion)
	{
		y = 23-epsilion;
	}
	x1 = static_cast <int> (floor(x));
	y1 = static_cast <int> (floor(y));
	x2 = x1+1;
	y2 = y1+1;

	//get x Position
	q11 = posX[(y1*32+ (31-x1))];
	q12 = posX[(y2*32 + (31-x1))];
	q21 = posX[(y1*32 + (31-x2))];
	q22 = posX[(y2*32 + (31-x2))];
	pC.x = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);

	//get y Position
	q11 = posY[(y1*32 + (31-x1))];
	q12 = posY[(y2*32 + (31-x1))];
	q21 = posY[(y1*32 + (31-x2))];
	q22 = posY[(y2*32 + (31-x2))];
	pC.y = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);

	return pC;
}


void makePos(float Z)
{
	for (int i = 0;i<768;i++)
	{	//negative value to adjust coordinate system according to image coord
		posX[i] = Z*tan(angleX[i] * PI / 180.0 );  
		posY[i] = -Z*tan(angleY[i] * PI / 180.0 );    
	}
	return;
}


int main(int argc, char* argv[])
{
	//Create look up table for given height (H-h)
	makePos(H-h);
	//std::cout << "Length of anglex = " << (sizeof(angleX)/sizeof(*angleX)) << std::endl;
	//std::cout << "Length of angley = " << (sizeof(angleY)/sizeof(*angleY)) << std::endl;
	/*
	for(int row = 0; row < 24; row++)
		{
			for(int col = 0; col < 32; col ++)
			{
				cout << angleX[(row*32 + (31-col))] << ",";
			}
			cout << endl << endl;
		}
		
		
	for(int row = 0; row < 24; row++)
	{
		for(int col = 0; col < 32; col ++)
		{
			cout << (int)posX[(row*32 + (31-col))] << ",";
		}
		cout << endl << endl;
	}
	
	for(int row = 0; row < 24; row++)
	{
		for(int col = 0; col < 32; col ++)
		{
			cout << (int)posY[(row*32 + (31-col))] << ",";
		}
		cout << endl << endl;
	}
	*/
	
	auto frame_time = std::chrono::microseconds(FRAME_TIME_MICROS + OFFSET_MICROS);
	
    Mat R_W_C(3,3,CV_32FC1);
    setIdentity(R_W_C);
    R_W_C.at<float>(0,0) = -1;
    R_W_C.at<float>(2,2) = -1;
    //cout << "Rotation Matrix " << endl << R_W_C << endl;
    
    Mat t_W_C(3,1,CV_32FC1);
    t_W_C.at<float>(0) = 0;
    t_W_C.at<float>(1) = YDIST;
    t_W_C.at<float>(2) = H;
    
	//cout << "translationVector " << endl << t_W_C << endl;


   if (wiringPiSetupGpio() < 0) return 1;  // Initializes wiringPi using the Broadcom GPIO pin numbers
   setAngle(90);
   pinMode(18, PWM_OUTPUT);
   pwmSetMode (PWM_MODE_MS);
   pwmSetClock(384); //clock at 50kHz (20us tick)
   pwmSetRange(1000); //range at 1000 ticks (20ms)
	    
    printf("Starting...\n");
    int success = 1;
    if (loadImg == 0)
    {
		while(success != 0)
		{
			success = configureMLX90640();
		}
	}
    pMOG2 = createBackgroundSubtractorMOG2(500,sigma*sigma,false); //MOG2 approach
    double learning_rate = -1;
    int frame_count = 0;

	VideoWriter out; // create video output
	
	if (makeVideo)
	{
		
		out.open("out.avi", CV_FOURCC('M','J','P','G'), FPS/2, Size(IMAGE_SCALE*COL+showThresh*IMAGE_SCALE*COL+showMorph*IMAGE_SCALE*COL, IMAGE_SCALE*ROW), true);	// set video output to 15 fps and MJPG
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
	
	if (saveMat) // empty folder
	{
		system("exec rm -r matCsv/imgFlip/*");
		system("exec rm -r matCsv/imgInt/*");
		system("exec rm -r matCsv/imgSmooth/*");
		system("exec rm -r matCsv/imgThresh/*");
		system("exec rm -r matCsv/imgMask/*");

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
	cout << "Calibrating ..." << endl;	
	
	while((char)keyboard != 'q' && (char)keyboard != 27){
		
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
			int i=0;
			while (success != 0)
			{
				success = getImg(img);
			}
		}

		if (saveImg)
		{
			char imgName[100];
			sprintf(imgName, "images/img_%06d.exr", frame_count);
			imwrite(imgName, img);
		}
		
		auto endTime = std::chrono::system_clock::now();
		if(!checkRange(img, true))
		{
			cout << " i contains NaN in frame: " << frame_count << endl;
			//cout << "img:" << endl << img << endl;
			//break; // exit while loop
		}
		minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
		//cout << "max at" << maxLoc << "with value " << maxVal << endl;
		//cout << "min at" << minLoc << "with value " << minVal << endl;
		//cout << "img (0,0)" << img.at<float>(0,0) <<  " img (23,0)" << img.at<float>(23,0) << " img (0,31): " << img.at<float>(0,31) << " img (23,31)" << img.at<float>(23,31) << endl;


		// Flip image around y-axis
		flip(img, imgFlip,1);
		
		
		// Apply cubic interpolation	
		//resize(imgInt, imgInt, Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), 0, 0, INTER_CUBIC);
		resize(imgFlip, imgInt, Size(IMAGE_SCALE*COL, IMAGE_SCALE*ROW), 0, 0, INTER_CUBIC);
		/*
		if(!checkRange(imgInt, true))
		{
			cout << " imgScale contains NaN" << endl;
			//cout << "img:" << endl << imgInt << endl;
			//break; // exit while loop
		}
		*/
		/*
		imgInt.convertTo(imgDebug,CV_8UC1);
		namedWindow( "imgInt", WINDOW_NORMAL);
		imshow("imgInt",imgDebug) ;
		*/
		/// Gaussian filtering
		GaussianBlur(imgInt, imgSmooth, Size(filterSize,filterSize),gaussSigma);
		/*
		imgSmooth.convertTo(imgDebug,CV_8UC1);
		namedWindow( "imgSmooth", WINDOW_NORMAL);
		imshow("imgSmooth", imgDebug);
		*/
		
		//cout << "Calibrating Background ..." << endl;		
		switch(BACKGROUNDMETHOD){
        case 1:	
			if (frame_count < BSIGNORE)
			{
				continue;
			}
			else if(frame_count < BSSAMPLE) 
			{
				accumulate(img,imgRaw);
				accumulateSquare(img, imgRaw2);
				
				accumulate(imgSmooth, acc);
				accumulateSquare(imgSmooth, sqacc);
				add(imgSmooth,imgBack,imgBack);
				continue;
			}
			else if (frame_count == BSSAMPLE)
			{
				imgBack = acc/(BSSAMPLE-BSIGNORE);
				// cout << "Background M" << imgBack << endl;
				Mat sqaccM = sqacc/(BSSAMPLE-BSIGNORE);
				Mat imgBack2 = imgBack.mul(imgBack);
				imgVar = sqaccM-imgBack2;
				sqrt(imgVar,imgStd);
				
				cout << "Calibration done" << endl;
				if (saveStdDev)
				{
					Mat imgRawMean = imgRaw/(BSSAMPLE-BSIGNORE);
					// cout << "Background M" << imgBack << endl;
					Mat imgRaw2Mean = imgRaw2/(BSSAMPLE-BSIGNORE);
					Mat imgRawMean2 = imgRawMean.mul(imgRawMean);
					Mat imgRawVar = imgRaw2Mean-imgRawMean2;
					sqrt(imgRawVar,imgRawStd);
				}				
				continue;
			}
			else
			{
				imgLevel = imgSmooth-imgBack-sigma*imgStd;			
				//cout << "After background substraction " << imgLevel << endl;
				threshold(imgLevel,imgThresh,0,1,0);
				//cout << "ImgThresh Depth " << imgThresh.type() << ",Rows " << imgThresh.cols << "Cols " << imgThresh.rows <<endl;
				//cout << "imgThresh " << imgThresh << endl;
				imgThresh.convertTo(imgThresh,CV_8UC1);
				//cout << "ImgThresh Depth " << imgThresh.type() << ",Rows " << imgThresh.cols << "Cols " << imgThresh.rows <<endl;
				//cout << "imgThresh " << imgThresh << endl;
				//imgSmooth.convertTo(imgSmooth,CV_8UC1);
				break;
			}
		
		 case 2: //Running Gaussian average
		
			if (frame_count < BSIGNORE)
			{
				continue;
			}
			else if(frame_count == BSIGNORE) 
			{
				imgMean == imgSmooth;  // Set I_0
				continue;
			}
			else if (frame_count == BSSAMPLE)
			{	
				rho = 0.00001; //reducing temporal window size
				cout << "Calibrating Done ..." << endl;
			}
			
			imgMean = rho*imgSmooth+(1-rho)*imgMean;
			absdiff(imgSmooth,imgMean,imgDist);
			imgDist.mul(imgDist);
			imgVar = imgDist*rho+(1-rho)*imgVar;
			sqrt(imgVar,imgStd);
			subtract(imgSmooth,imgMean,imgSub);
			divide(imgSub,imgStd,imgZScore); //  confidence interval
			
			
			//cout << "Z-Score" << endl << endl << imgZScore;
			threshold(imgZScore,imgThresh,kThresh,1,THRESH_BINARY);
			imgThresh.convertTo(imgThresh,CV_8UC1);
			//tempVal = mean( imgDist );
			//cout << "mean of imgDist :" << tempVal.val[0] << "at frame" << frame_count <<endl;
			//cout << "eTA: " << eTa << endl;

			//tempVal = mean( imgStd );
			//cout << "mean of imgStd :" << tempVal.val[0] << "at frame" << frame_count <<endl;
							
			//cout << "imgMean"<< endl << imgMean<< endl;
			
			if (frame_count < BSSAMPLE)
			{
				continue;
			}
			else 
			{
				break;
			}

			
        case 3:
			pMOG2->apply(m*imgSmooth+b,imgThresh,learning_rate);
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

		/// Apply closing morphology operation
		 morphologyEx( imgThresh, imgMorph, 1 , element );
		 /// Apply opening morphology operation
		 morphologyEx( imgMorph, imgMorph, 0 , element );

		
		// find contours
		findContours(imgMorph, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		// sort contours starting with biggest contour
		sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Point2f>pointList;
		vector<Point2f>centroidList;
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
				centroidList.push_back(pC);
				
				//Second Approach
				imgMask = Mat::zeros(imgSmooth.size(), CV_8UC1);
				drawContours(imgMask,contours,i,Scalar(1),FILLED);  
				Mat masked(imgSmooth.size(),CV_8UC1,Scalar(0));
				imgSmooth.copyTo(masked,imgMask);
				Moments m = moments(masked,false);
				Point2d p(m.m10/m.m00, m.m01/m.m00); //u0, v0
				pointList.push_back(p);
		    }
		 }
		 
		if (saveMat)
		{
			char csvName[100];
			sprintf(csvName, "matCsv/imgFlip/img_%06d.csv", frame_count);
			writeCSV(csvName, imgFlip);
			sprintf(csvName, "matCsv/imgInt/img_%06d.csv", frame_count);
			writeCSV(csvName, imgInt);
			sprintf(csvName, "matCsv/imgSmooth/img_%06d.csv", frame_count);
			writeCSV(csvName, imgSmooth);
			sprintf(csvName, "matCsv/imgThresh/img_%06d.csv", frame_count);
			writeCSV(csvName, imgThresh);
			sprintf(csvName, "matCsv/imgMask/img_%06d.csv", frame_count);
			writeCSV(csvName, imgMask);
		}
		
		
		if (makeImage)
			{
			// ClearScreen();
			imgSmooth = m*imgSmooth+b;
			imgSmooth.convertTo(imgSmooth,CV_8UC1);
			applyColorMap(imgSmooth, imgColor, COLORMAP_JET); // 1630 us  

			for( int i = 0; i< pointList.size(); i++ ) // iterate through each contour.
			{
				imgColor.convertTo(imgColor,CV_8UC1);
				drawMarker(imgColor,pointList[i],Scalar(0,0,0),0,1*IMAGE_SCALE,1*IMAGE_SCALE,8);
				putText(imgColor, to_string(i+1), pointList[i], FONT_HERSHEY_PLAIN,0.5*IMAGE_SCALE,  Scalar(0,0,0));
			}
		}
		
		
	    // show the resultant image
        if(showImage)
        {	
			if(showMorph)
			{
				imgMorph = 255*imgMorph;
				imgMorph.convertTo(imgMorph,CV_8UC1);
				cvtColor(imgMorph, imgMorph, COLOR_GRAY2BGR);
				hconcat(imgMorph,imgColor,imgColor);
			}
			
		    if(showThresh)
		    {
				imgThresh = 255*imgThresh;
				cvtColor(imgThresh, imgThresh, COLOR_GRAY2BGR);
				hconcat(imgThresh,imgColor,imgColor);
			}
			
			// show the resultant image
			namedWindow( "Contours", WINDOW_NORMAL);
			imshow("Contours", imgColor);
			waitKey(1);
		}
		
		if (makeVideo)	out << imgColor; // add frame to video

		
		for( int i = 0; i< centroidList.size(); i++ ) // iterate through each contour. 
		{
			//transfer to world frame
			camPnt.at<float>(0) = centroidList[i].x; //u
			camPnt.at<float>(1) = centroidList[i].y; //v
			camPnt.at<float>(2) = H-h;
			worldPnt = R_W_C*camPnt+t_W_C;
			//cout << i+1 <<".Human in  World coordinates" << endl << worldPnt << endl;	
			cout << i+1 <<".Human in  Camera coordinates" << endl << camPnt << endl;	
			float theta = atan2(worldPnt.at<float>(0),worldPnt.at<float>(1))*180/PI; //atan2(y,x)
			//cout << "Theta" << theta << endl;
			//cout << "Theta servo"  << 90-theta <<endl;
				
				if (runServo && i == 0) 
				{
					setAngle(90-theta);
				}
				
				if (saveTrajectory && i == 0)
				{	
					trajectory << frame_count << ",";
					trajectory << camPnt.at<float>(0) << ",";
					trajectory << camPnt.at<float>(1) << ",";
					trajectory << camPnt.at<float>(2) << ",";
					trajectory << worldPnt.at<float>(0) << ",";
					trajectory << worldPnt.at<float>(1) << ",";
					trajectory << worldPnt.at<float>(2) << ",";
					trajectory << theta <<  ",";			
					trajectory << endl;
				}
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
		

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
		//cout << "Measured time " << elapsedTime.count() << endl;
		//cout << "Percentage of calulated time " << 100.0/elapsed.count()*elapsedTime.count()<< endl;
		
		if (frame_time.count()-elapsed.count() < 0)
		{
			
			/*
			 * cout << frame_count << "th Frame not ready" << endl;
			
			 /* cout << "Take smaller FPS" << endl;
			cout << "Frame time" << frame_time.count()<< " us" << endl;    
			cout << "Elaped time" << elapsed.count()<< " us" << endl;    
			cout << "Wait time" << frame_time.count()-elapsed.count()<< " us" << endl; 
			*/
		}
		if (makeVideo)
		{
			this_thread::sleep_for(chrono::microseconds(frame_time - elapsed));	
		}
	}
	if (saveTrajectory)	trajectory.close();
	//destroy GUI windows
    destroyAllWindows();
    cout <<"imgStd" << endl << imgStd << endl;	
    return EXIT_SUCCESS;
    return 0;
}




