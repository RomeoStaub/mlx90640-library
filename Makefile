I2C_MODE = RPI
I2C_LIBS = -lbcm2835

LIBS_opencv = $(shell pkg-config --libs --cflags opencv) 

LDLIBS    = -lwiringPi -lwiringPiDev -lpthread -lm -lcrypt -lrt

all: examples

examples: debugBlob blob

libMLX90640_API.so: functions/MLX90640_API.o functions/MLX90640_$(I2C_MODE)_I2C_Driver.o
	$(CXX) -fPIC -shared $^ -o $@ $(I2C_LIBS)

libMLX90640_API.a: functions/MLX90640_API.o functions/MLX90640_$(I2C_MODE)_I2C_Driver.o
	ar rcs $@ $^
	ranlib $@

functions/MLX90640_API.o functions/MLX90640_RPI_I2C_Driver.o functions/MLX90640_LINUX_I2C_Driver.o : CXXFLAGS+=-fPIC -I headers -shared $(I2C_LIBS) 
examples/debugBlob.o  examples/blob.o : CXXFLAGS+=-std=c++11 $(shell pkg-config --cflags opencv)

debugBlob blob: CXXFLAGS+=-I. -std=c++11

debugBlob: examples/debugBlob.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) $(LIBS_opencv) $(LDLIBS)

blob: examples/blob.o libMLX90640_API.a
	$(CXX) -L/home/pi/mlx90640-library $^ -o $@ $(I2C_LIBS) $(LIBS_opencv)  

bcm2835-1.55.tar.gz:	
	wget http://www.airspayce.com/mikem/bcm2835/bcm2835-1.55.tar.gz

bcm2835-1.55: bcm2835-1.55.tar.gz
	tar xzvf bcm2835-1.55.tar.gz

bcm2835: bcm2835-1.55
	cd bcm2835-1.55; ./configure; make; sudo make install

clean:
	rm -f debugBlob blob
	rm -f examples/*.o
	rm -f examples/lib/*.o
	rm -f functions/*.o
	rm -f *.o
	rm -f *.so
	rm -f test
	rm -f *.a
