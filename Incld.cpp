#include "Incld.h"




void Command(string command, int msg) //makes process
{
	FILE *fp;
	int status;
	fp=popen(command.c_str(),"w");
	status=pclose(fp);
	if(msg>0)
	{
		cout<<status<<endl;
	}
	
	return;
}




double conv(string q) //string->double
{
	stringstream ss;
	double val;
	ss<<q;
	ss>>val;
	return val;
}




string conv(double q) //double->string
{
	ostringstream strs;
	strs <<q;
	string str = strs.str();
	return str;
}







vector<string> Divide_string(string text, string delimiter) //dividing string based on delimiter
{
    vector<string> divided;
    string onestr,oneletter;
    for(int i=0;i<text.size();++i)
    {
        oneletter=text[i];
        if(i!=0 and oneletter.compare(delimiter)==0) //delimiter found
        {
            divided.push_back(onestr);
            onestr.clear();
        }
        else
        {
            onestr+=text[i];
            if(i==text.size()-1){divided.push_back(onestr);}
        }
    }
    return divided;
}









//getting parameter value(s) from file, based on 1st column called par_name
vector<string> Get_parameter(string fname, string par_name) 
{
    string line,thisparname,thispar_entry;
    vector<string> thispar,par; //storing parameter (could contain more entries than one)
    ifstream f(fname.c_str());
    while(!f.eof())
    {
        getline(f,line);
        thispar.clear();
		if(line.size()<2){continue;}
        thispar=Divide_string(line," "); //dividing the entry
        thisparname=thispar[0];
        if(thisparname.compare(par_name)==0) //parameter found
        {
            for(int i=1;i<thispar.size();++i) //loop over divided parts
            {
                thispar_entry=thispar[i];
                if(thispar_entry[0]=='#'){break;} //values/words beginning form # are treating as comments
                else par.push_back(thispar_entry);
            }
            break;
        }
    }
    f.close();
    return par;
}









//packing real, imag arrays into cufft-working one
__global__ void pack_arr_cufft(const double* real, const double* imag, cufftDoubleComplex* complex_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
	{
        complex_out[idx].x = real[idx];
        complex_out[idx].y = imag[idx];
    }
}






//unpacking back into real,imag arrays
__global__ void unpack_arr_cufft(const cufftDoubleComplex* complex_in, double* real, double* imag, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
	{
        real[idx] = complex_in[idx].x;
        imag[idx] = complex_in[idx].y;
    }
}





//normalizing inverse fft output
__global__ void normalize_invfft(cufftDoubleComplex* data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
	{
        data[idx].x /= N;
        data[idx].y /= N;
    }
}





void FFT2D_flat(double *img_real, double *img_imag, int s, int direction)
{
    const int N = s * s;

    //allocate memory
    double* d_real;
    double* d_imag;
    cudaMalloc((void**)&d_real, sizeof(double) * N);
    cudaMalloc((void**)&d_imag, sizeof(double) * N);

    // copy data to GPU
    cudaMemcpy(d_real, img_real, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, img_imag, sizeof(double) * N, cudaMemcpyHostToDevice);

    //create plan
    cufftHandle plan;
    cufftResult result = cufftPlan2d(&plan, s, s, CUFFT_Z2Z); // Z2Z = double complex
    if (result != CUFFT_SUCCESS)
	{
        cerr <<"[Error]: failed creating cuFFT plan:("<<endl;
        return;
    }
	

    // packing for cuFFT function:
    cufftDoubleComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftDoubleComplex) * N);
	
	//1-D threads-grid
	int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	pack_arr_cufft<<<blocks, threadsPerBlock>>>(d_real, d_imag, d_data, N);
	cudaDeviceSynchronize();
	
	
    //running FFT
	if(direction==1)
	{
		cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
	}
	else
	{
		cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE);
		
		normalize_invfft<<<blocks, threadsPerBlock>>>(d_data, N);
		cudaDeviceSynchronize();
	}
    

	//unpacking back to real and imaginary arrays
	unpack_arr_cufft<<<blocks, threadsPerBlock>>>(d_data, d_real, d_imag, N);
	cudaDeviceSynchronize();
	
	
	// copy data back to CPU
    cudaMemcpy(img_real,d_real, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_imag,d_imag, sizeof(double) * N, cudaMemcpyDeviceToHost);


    cufftDestroy(plan);
    cudaFree(d_real);
    cudaFree(d_imag);
    cudaFree(d_data);
    return;
}




