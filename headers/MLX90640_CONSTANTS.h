// Constants.h
#if !defined(MYLIB_CONSTANTS)
#define MLX_I2C_ADDR 0x33
#define LOWBND 22
#define UPBND 35
#define IMAGE_SCALE 2.0
#define ROW 24
#define COL 32
#define H 2500 //ceiling height in mm
#define h 0  //subject height in mm
#define FOVX 100.0
#define FOVY 75.0
#define PI 3.14159265
//#define YDIST 1000
#define YDIST 0
#define MINAREA 0 //minimum amount of pixels to detect human
#define BSSAMPLE 500
#define BSIGNORE 20

/// Servo Parameters
#define SVLOW 20  //servo command at 0 degree
#define SVHIGH 110 //servo command at 180 degree


// Valid frame rates are 1, 2, 4, 8, 16, 32
// The i2c baudrate is set to 1mhz to support these
#define FPS 32
#define FRAME_TIME_MICROS (1000000/FPS*2)

// Despite the framerate being ostensibly FPS hz
// The frame is often not ready in time
// This offset is added to the FRAME_TIME_MICROS
// to account for this.
#define OFFSET_MICROS 3000

// Choose Backgroundsubstraction method
// 1 =  Substract average of first 30
// 2 =  MOG2 Background subtractor
#define BACKGROUNDMETHOD 1

float angleX[] = {50.4,47.2,42.6,39.6,35.3,32.5,28.4,25.7,21.9,19.4,15.7,13.3,9.8,7.4,4.0,1.7,-1.7,-4.0,-7.4,-9.8,-13.3,-15.7,-19.4,-21.9,-25.7,-28.4,-32.5,-35.3,-39.6,-42.6,-47.2,-50.4,49.9,46.7,42.2,39.2,35.0,32.1,28.1,25.5,21.7,19.1,15.6,13.1,9.7,7.3,4.0,1.7,-1.7,-4.0,-7.3,-9.7,-13.1,-15.6,-19.1,-21.7,-25.5,-28.1,-32.1,-35.0,-39.2,-42.2,-46.7,-49.9,49.4,46.2,41.7,38.7,34.5,31.7,27.7,25.1,21.3,18.8,15.3,12.9,9.5,7.2,3.9,1.6,-1.6,-3.9,-7.2,-9.5,-12.9,-15.3,-18.8,-21.3,-25.1,-27.7,-31.7,-34.5,-38.7,-41.7,-46.2,-49.4,49.0,45.8,41.4,38.4,34.2,31.4,27.5,24.8,21.1,18.6,15.1,12.7,9.4,7.1,3.8,1.6,-1.6,-3.8,-7.1,-9.4,-12.7,-15.1,-18.6,-21.1,-24.8,-27.5,-31.4,-34.2,-38.4,-41.4,-45.8,-49.0,48.5,45.4,40.9,38.0,33.8,31.0,27.1,24.5,20.8,18.3,14.9,12.5,9.2,7.0,3.8,1.6,-1.6,-3.8,-7.0,-9.2,-12.5,-14.9,-18.3,-20.8,-24.5,-27.1,-31.0,-33.8,-38.0,-40.9,-45.4,-48.5,48.2,45.1,40.6,37.7,33.5,30.8,26.9,24.3,20.6,18.2,14.7,12.4,9.1,6.9,3.7,1.6,-1.6,-3.7,-6.9,-9.1,-12.4,-14.7,-18.2,-20.6,-24.3,-26.9,-30.8,-33.5,-37.7,-40.6,-45.1,-48.2,47.9,44.7,40.3,37.4,33.2,30.5,26.6,24.0,20.4,17.9,14.5,12.2,9.0,6.8,3.7,1.5,-1.5,-3.7,-6.8,-9.0,-12.2,-14.5,-17.9,-20.4,-24.0,-26.6,-30.5,-33.2,-37.4,-40.3,-44.7,-47.9,47.7,44.5,40.1,37.2,33.0,30.3,26.4,23.8,20.2,17.8,14.4,12.1,8.9,6.7,3.6,1.5,-1.5,-3.6,-6.7,-8.9,-12.1,-14.4,-17.8,-20.2,-23.8,-26.4,-30.3,-33.0,-37.2,-40.1,-44.5,-47.7,47.4,44.3,39.9,36.9,32.8,30.1,26.2,23.6,20.0,17.6,14.2,11.9,8.7,6.6,3.5,1.5,-1.5,-3.5,-6.6,-8.7,-11.9,-14.2,-17.6,-20.0,-23.6,-26.2,-30.1,-32.8,-36.9,-39.9,-44.3,-47.4,47.3,44.1,39.7,36.8,32.7,30.0,26.1,23.5,19.9,17.5,14.1,11.9,8.7,6.5,3.5,1.5,-1.5,-3.5,-6.5,-8.7,-11.9,-14.1,-17.5,-19.9,-23.5,-26.1,-30.0,-32.7,-36.8,-39.7,-44.1,-47.3,47.2,44.0,39.6,36.7,32.6,29.8,26.0,23.4,19.8,17.4,14.0,11.8,8.6,6.5,3.5,1.4,-1.4,-3.5,-6.5,-8.6,-11.8,-14.0,-17.4,-19.8,-23.4,-26.0,-29.8,-32.6,-36.7,-39.6,-44.0,-47.2,47.1,44.0,39.6,36.7,32.5,29.8,26.0,23.4,19.8,17.4,14.0,11.7,8.6,6.4,3.4,1.4,-1.4,-3.4,-6.4,-8.6,-11.7,-14.0,-17.4,-19.8,-23.4,-26.0,-29.8,-32.5,-36.7,-39.6,-44.0,-47.1,47.1,44.0,39.6,36.7,32.5,29.8,26.0,23.4,19.8,17.4,14.0,11.7,8.6,6.4,3.4,1.4,-1.4,-3.4,-6.4,-8.6,-11.7,-14.0,-17.4,-19.8,-23.4,-26.0,-29.8,-32.5,-36.7,-39.6,-44.0,-47.1,47.2,44.0,39.6,36.7,32.6,29.8,26.0,23.4,19.8,17.4,14.0,11.8,8.6,6.5,3.5,1.4,-1.4,-3.5,-6.5,-8.6,-11.8,-14.0,-17.4,-19.8,-23.4,-26.0,-29.8,-32.6,-36.7,-39.6,-44.0,-47.2,47.3,44.1,39.7,36.8,32.7,30.0,26.1,23.5,19.9,17.5,14.1,11.9,8.7,6.5,3.5,1.5,-1.5,-3.5,-6.5,-8.7,-11.9,-14.1,-17.5,-19.9,-23.5,-26.1,-30.0,-32.7,-36.8,-39.7,-44.1,-47.3,47.4,44.3,39.9,36.9,32.8,30.1,26.2,23.6,20.0,17.6,14.2,11.9,8.7,6.6,3.5,1.5,-1.5,-3.5,-6.6,-8.7,-11.9,-14.2,-17.6,-20.0,-23.6,-26.2,-30.1,-32.8,-36.9,-39.9,-44.3,-47.4,47.7,44.5,40.1,37.2,33.0,30.3,26.4,23.8,20.2,17.8,14.4,12.1,8.9,6.7,3.6,1.5,-1.5,-3.6,-6.7,-8.9,-12.1,-14.4,-17.8,-20.2,-23.8,-26.4,-30.3,-33.0,-37.2,-40.1,-44.5,-47.7,47.9,44.7,40.3,37.4,33.2,30.5,26.6,24.0,20.4,17.9,14.5,12.2,9.0,6.8,3.7,1.5,-1.5,-3.7,-6.8,-9.0,-12.2,-14.5,-17.9,-20.4,-24.0,-26.6,-30.5,-33.2,-37.4,-40.3,-44.7,-47.9,48.2,45.1,40.6,37.7,33.5,30.8,26.9,24.3,20.6,18.2,14.7,12.4,9.1,6.9,3.7,1.6,-1.6,-3.7,-6.9,-9.1,-12.4,-14.7,-18.2,-20.6,-24.3,-26.9,-30.8,-33.5,-37.7,-40.6,-45.1,-48.2,48.5,45.4,40.9,38.0,33.8,31.0,27.1,24.5,20.8,18.3,14.9,12.5,9.2,7.0,3.8,1.6,-1.6,-3.8,-7.0,-9.2,-12.5,-14.9,-18.3,-20.8,-24.5,-27.1,-31.0,-33.8,-38.0,-40.9,-45.4,-48.5,49.0,45.8,41.4,38.4,34.2,31.4,27.5,24.8,21.1,18.6,15.1,12.7,9.4,7.1,3.8,1.6,-1.6,-3.8,-7.1,-9.4,-12.7,-15.1,-18.6,-21.1,-24.8,-27.5,-31.4,-34.2,-38.4,-41.4,-45.8,-49.0,49.4,46.2,41.7,38.7,34.5,31.7,27.7,25.1,21.3,18.8,15.3,12.9,9.5,7.2,3.9,1.6,-1.6,-3.9,-7.2,-9.5,-12.9,-15.3,-18.8,-21.3,-25.1,-27.7,-31.7,-34.5,-38.7,-41.7,-46.2,-49.4,49.9,46.7,42.2,39.2,35.0,32.1,28.1,25.5,21.7,19.1,15.6,13.1,9.7,7.3,4.0,1.7,-1.7,-4.0,-7.3,-9.7,-13.1,-15.6,-19.1,-21.7,-25.5,-28.1,-32.1,-35.0,-39.2,-42.2,-46.7,-49.9,50.4,47.2,42.6,39.6,35.3,32.5,28.4,25.7,21.9,19.4,15.7,13.3,9.8,7.4,4.0,1.7,-1.7,-4.0,-7.4,-9.8,-13.3,-15.7,-19.4,-21.9,-25.7,-28.4,-32.5,-35.3,-39.6,-42.6,-47.2,-50.4};
float angleY[] = {37.3,36.9,36.3,35.9,35.3,35.0,34.5,34.2,33.8,33.5,33.2,33.0,32.8,32.7,32.6,32.5,32.5,32.6,32.7,32.8,33.0,33.2,33.5,33.8,34.2,34.5,35.0,35.3,35.9,36.3,36.9,37.3,34.3,33.9,33.4,33.0,32.5,32.1,31.7,31.4,31.0,30.8,30.5,30.3,30.1,30.0,29.8,29.8,29.8,29.8,30.0,30.1,30.3,30.5,30.8,31.0,31.4,31.7,32.1,32.5,33.0,33.4,33.9,34.3,30.1,29.8,29.3,28.9,28.4,28.1,27.7,27.5,27.1,26.9,26.6,26.4,26.2,26.1,26.0,26.0,26.0,26.0,26.1,26.2,26.4,26.6,26.9,27.1,27.5,27.7,28.1,28.4,28.9,29.3,29.8,30.1,27.3,27.0,26.5,26.2,25.7,25.5,25.1,24.8,24.5,24.3,24.0,23.8,23.6,23.5,23.4,23.4,23.4,23.4,23.5,23.6,23.8,24.0,24.3,24.5,24.8,25.1,25.5,25.7,26.2,26.5,27.0,27.3,23.3,23.0,22.6,22.3,21.9,21.7,21.3,21.1,20.8,20.6,20.4,20.2,20.0,19.9,19.8,19.8,19.8,19.8,19.9,20.0,20.2,20.4,20.6,20.8,21.1,21.3,21.7,21.9,22.3,22.6,23.0,23.3,20.6,20.4,20.0,19.7,19.4,19.1,18.8,18.6,18.3,18.2,17.9,17.8,17.6,17.5,17.4,17.4,17.4,17.4,17.5,17.6,17.8,17.9,18.2,18.3,18.6,18.8,19.1,19.4,19.7,20.0,20.4,20.6,16.8,16.6,16.3,16.0,15.7,15.6,15.3,15.1,14.9,14.7,14.5,14.4,14.2,14.1,14.0,14.0,14.0,14.0,14.1,14.2,14.4,14.5,14.7,14.9,15.1,15.3,15.6,15.7,16.0,16.3,16.6,16.8,14.2,14.0,13.7,13.6,13.3,13.1,12.9,12.7,12.5,12.4,12.2,12.1,11.9,11.9,11.8,11.7,11.7,11.8,11.9,11.9,12.1,12.2,12.4,12.5,12.7,12.9,13.1,13.3,13.6,13.7,14.0,14.2,10.5,10.3,10.1,10.0,9.8,9.7,9.5,9.4,9.2,9.1,9.0,8.9,8.7,8.7,8.6,8.6,8.6,8.6,8.7,8.7,8.9,9.0,9.1,9.2,9.4,9.5,9.7,9.8,10.0,10.1,10.3,10.5,8.0,7.8,7.7,7.6,7.4,7.3,7.2,7.1,7.0,6.9,6.8,6.7,6.6,6.5,6.5,6.4,6.4,6.5,6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.6,7.7,7.8,8.0,4.3,4.2,4.2,4.1,4.0,4.0,3.9,3.8,3.8,3.7,3.7,3.6,3.5,3.5,3.5,3.4,3.4,3.5,3.5,3.5,3.6,3.7,3.7,3.8,3.8,3.9,4.0,4.0,4.1,4.2,4.2,4.3,1.8,1.8,1.7,1.7,1.7,1.7,1.6,1.6,1.6,1.6,1.5,1.5,1.5,1.5,1.4,1.4,1.4,1.4,1.5,1.5,1.5,1.5,1.6,1.6,1.6,1.6,1.7,1.7,1.7,1.7,1.8,1.8,-1.8,-1.8,-1.7,-1.7,-1.7,-1.7,-1.6,-1.6,-1.6,-1.6,-1.5,-1.5,-1.5,-1.5,-1.4,-1.4,-1.4,-1.4,-1.5,-1.5,-1.5,-1.5,-1.6,-1.6,-1.6,-1.6,-1.7,-1.7,-1.7,-1.7,-1.8,-1.8,-4.3,-4.2,-4.2,-4.1,-4.0,-4.0,-3.9,-3.8,-3.8,-3.7,-3.7,-3.6,-3.5,-3.5,-3.5,-3.4,-3.4,-3.5,-3.5,-3.5,-3.6,-3.7,-3.7,-3.8,-3.8,-3.9,-4.0,-4.0,-4.1,-4.2,-4.2,-4.3,-8.0,-7.8,-7.7,-7.6,-7.4,-7.3,-7.2,-7.1,-7.0,-6.9,-6.8,-6.7,-6.6,-6.5,-6.5,-6.4,-6.4,-6.5,-6.5,-6.6,-6.7,-6.8,-6.9,-7.0,-7.1,-7.2,-7.3,-7.4,-7.6,-7.7,-7.8,-8.0,-10.5,-10.3,-10.1,-10.0,-9.8,-9.7,-9.5,-9.4,-9.2,-9.1,-9.0,-8.9,-8.7,-8.7,-8.6,-8.6,-8.6,-8.6,-8.7,-8.7,-8.9,-9.0,-9.1,-9.2,-9.4,-9.5,-9.7,-9.8,-10.0,-10.1,-10.3,-10.5,-14.2,-14.0,-13.7,-13.6,-13.3,-13.1,-12.9,-12.7,-12.5,-12.4,-12.2,-12.1,-11.9,-11.9,-11.8,-11.7,-11.7,-11.8,-11.9,-11.9,-12.1,-12.2,-12.4,-12.5,-12.7,-12.9,-13.1,-13.3,-13.6,-13.7,-14.0,-14.2,-16.8,-16.6,-16.3,-16.0,-15.7,-15.6,-15.3,-15.1,-14.9,-14.7,-14.5,-14.4,-14.2,-14.1,-14.0,-14.0,-14.0,-14.0,-14.1,-14.2,-14.4,-14.5,-14.7,-14.9,-15.1,-15.3,-15.6,-15.7,-16.0,-16.3,-16.6,-16.8,-20.6,-20.4,-20.0,-19.7,-19.4,-19.1,-18.8,-18.6,-18.3,-18.2,-17.9,-17.8,-17.6,-17.5,-17.4,-17.4,-17.4,-17.4,-17.5,-17.6,-17.8,-17.9,-18.2,-18.3,-18.6,-18.8,-19.1,-19.4,-19.7,-20.0,-20.4,-20.6,-23.3,-23.0,-22.6,-22.3,-21.9,-21.7,-21.3,-21.1,-20.8,-20.6,-20.4,-20.2,-20.0,-19.9,-19.8,-19.8,-19.8,-19.8,-19.9,-20.0,-20.2,-20.4,-20.6,-20.8,-21.1,-21.3,-21.7,-21.9,-22.3,-22.6,-23.0,-23.3,-27.3,-27.0,-26.5,-26.2,-25.7,-25.5,-25.1,-24.8,-24.5,-24.3,-24.0,-23.8,-23.6,-23.5,-23.4,-23.4,-23.4,-23.4,-23.5,-23.6,-23.8,-24.0,-24.3,-24.5,-24.8,-25.1,-25.5,-25.7,-26.2,-26.5,-27.0,-27.3,-30.1,-29.8,-29.3,-28.9,-28.4,-28.1,-27.7,-27.5,-27.1,-26.9,-26.6,-26.4,-26.2,-26.1,-26.0,-26.0,-26.0,-26.0,-26.1,-26.2,-26.4,-26.6,-26.9,-27.1,-27.5,-27.7,-28.1,-28.4,-28.9,-29.3,-29.8,-30.1,-34.3,-33.9,-33.4,-33.0,-32.5,-32.1,-31.7,-31.4,-31.0,-30.8,-30.5,-30.3,-30.1,-30.0,-29.8,-29.8,-29.8,-29.8,-30.0,-30.1,-30.3,-30.5,-30.8,-31.0,-31.4,-31.7,-32.1,-32.5,-33.0,-33.4,-33.9,-34.3,-37.3,-36.9,-36.3,-35.9,-35.3,-35.0,-34.5,-34.2,-33.8,-33.5,-33.2,-33.0,-32.8,-32.7,-32.6,-32.5,-32.5,-32.6,-32.7,-32.8,-33.0,-33.2,-33.5,-33.8,-34.2,-34.5,-35.0,-35.3,-35.9,-36.3,-36.9,-37.3};
float posX[ROW*COL];
float posY[ROW*COL];
#endif