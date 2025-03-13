#compiler and options
CC=nvcc
OPTIONS=-x cu --expt-relaxed-constexpr --std=c++14 -lcuda `pkg-config --cflags --libs opencv4`

#output executable
TARGET=GField.exe

#files to compile:
SRC = GField.cpp Incld.cpp


#compiling:
$(TARGET): GField.cpp
	$(CC) $(OPTIONS) $(SRC) -o $(TARGET)