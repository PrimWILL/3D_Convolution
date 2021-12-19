CC=g++
CC2=nvcc
CFLAGS=-g -Wall
OBJS=convavx.o
TARGET=convavx
TARGET2=convgpu

all: $(TARGET) $(TARGET2)

$(TARGET) : convavx.cpp
	$(CC) -mavx2 -o $(TARGET) convavx.cpp -pthread

$(TARGET2) : convgpu.cu
	$(CC2) convgpu.cu -o $(TARGET2)

clean:
	rm -f *.o
	rm -f $(TARGET)
	rm -f $(TARGET2)
