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


void Command(string command, int msg=0); //makes process


double conv(string q); //string->double


string conv(double q); //double->string


vector<string> Divide_string(string text, string delimiter); //dividing string based on delimiter


//getting parameter value(s) from file, based on 1st column called par_name
vector<string> Get_parameter(string fname, string par_name);


//1D FFT
int FFT(int dir,int m,double *x,double *y);


//2D FFT: (in_real,in_imag)->FFT(in_real,in_imag); real,imag of size [s*s]
int FFT2D_flat(double *img_real, double *img_imag, int s, int direction);







template <typename T> //reading columns from file to list of arrays - with fname as parameter
void Fread(string fname, initializer_list<T*> arrays, initializer_list<int> cols, int array_size)
{
    ifstream f(fname.c_str());
    if (!f.is_open())
	{
        cerr<<"Error opening file: "<<fname<<endl;
        return;
    }

    vector<int> col_indices(cols.begin(), cols.end());
    int nc_selected = col_indices.size();

    // Ensure that arrays and cols sizes match
    if (arrays.size()!= nc_selected)
	{
        cerr<<"Error: mismatch between number of vectors and columns :("<<endl;
        return;
    }

    vector<T*> array_ptrs(arrays); //convert initializer_list to vector
    T value;
    string line;
    int row=0,current_col,selected_index;

    while (getline(f, line) && row < array_size)
	{
        istringstream iss(line);
        current_col = 0;
        selected_index = 0;

        for (current_col=0;iss>>value;++current_col)
		{
            //whether current_col matches desired one
            if (selected_index < nc_selected && current_col == col_indices[selected_index])
			{
                array_ptrs[selected_index][row]=value;
                ++selected_index;
            }
        }
        ++row;
    }

    f.close();
	return;
}





#endif