/*
FFT functions are based on: https://paulbourke.net/miscellaneous/dft/
All other functions written by Paweł Drozda

*/


#ifndef INCDL_H
#define INCDL_H

#include<iostream>
#include<fstream>
#include<time.h>
#include<string>
#include<vector>
#include <initializer_list>  //using array in fucntion args without pre-declaration

//#include <filesystem>
#include <opencv2/opencv.hpp>
//#include<dirent.h> //for reading files in dir without filesystem
//#include <sys/stat.h> //for mkdir
//#include<sys/types.h>
#define BLOCK_SIZE 8  //block size of GPU computation
#include <curand_kernel.h> //for random in CUDA
#include<algorithm>

using namespace std;


void Command(string command); //makes process


double conv(string q); //string->double


string conv(double q); //double->string


vector<string> Divide_string(string text, string delimiter); //dividing string based on delimiter


//getting parameter value(s) from file, based on 1st column called par_name
vector<string> Get_parameter(string fname, string par_name);


//1D FFT
int FFT(int dir,int m,double *x,double *y);


//2D FFT: (in_real,in_imag)->FFT(in_real,in_imag); real,imag of size [s*s]
int FFT2D_flat(double *img_real, double *img_imag, int s, int direction);



#endif