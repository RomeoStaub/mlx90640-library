# mlx90640-library
MLX90640 library functions for PATHOS based on :/github.com/pimoroni/mlx90640-library


## Raspberry Pi Users

### BCM2835 Library Mode

To use the bcm2835 library, install like so:


```text
make bcm2835
```

Or, step by step:

```text
wget http://www.airspayce.com/mikem/bcm2835/bcm2835-1.55.tar.gz
tar xvfz bcm2835-1.55.tar.gz
cd bcm2835-1.55
./configure
make
sudo make install
```

To install dependencies:

```text
sudo apt-get install libavutil-dev libavcodec-dev libavformat-dev
```

Then just `make` and `sudo ./blob` or  `sudo ./debugBlob` as listed below:

# debugBlob

```
sudo ./debugBlob
```

This example helps for debugging.
If you would like to see the output image, set "IMAGE_SCALE" to a smaller number.
To change the frame per seconds, set "FPS" to 1,2,4,8, or 16




# blob

```
sudo ./blob
```

This example 
