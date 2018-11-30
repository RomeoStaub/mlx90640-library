#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include "headers/MLX90640_API.h"
#include <opencv2/opencv.hpp>


#define MLX_I2C_ADDR 0x33
using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat im;
    im = imread( argv[1], 1 );

    if ( !im.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(imgray, 127, 255, 0)
	im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return 0;
}
