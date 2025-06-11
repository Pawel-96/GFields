//*************************************************************************************************************************************//
//                                                                                                                                     //
// This file is part of GField, written by Paweł Drozda, except of Fast Fourier Transform functions which are based on:                //
// http://paulbourke.net/miscellaneous/dft/                                                                                            //
//                                                                                                                                     //
// To the extent possible under law, the author has dedicated this software	                                                       //
// to the public domain under the Creative Commons CC0 1.0 Universal License.                                                          //
//                                                                                                                                     //
// GField is distributed in the hope that it will be useful, but                                                                       //
// WITHOUT ANY WARRANTY; without even the implied warranty of                                                                          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                                                                //
//                                                                                                                                     //
// You should have received a copy of the CC0-1.0 License                                                                              //
// along with GField.  If not, see <ttps://creativecommons.org/publicdomain/zero/1.0/>.                                                //
//                                                                                                                                     //
//                                                                                                                                     //
//*************************************************************************************************************************************//



#include "Incld.h"


const string paramfile="param.txt";
const string pkfile="Pk.txt";

const int log2size=conv(Get_parameter(paramfile,"log2size")[0]); //s=2^log2size
const int probing=conv(Get_parameter(paramfile,"Pk_probing")[0]); //listed power spectrum probing
const string fout=Get_parameter(paramfile,"fname_out")[0]; //output filename
const string fout_img=Get_parameter(paramfile,"fname_img")[0]; //output filename
const string Pkform=Get_parameter(paramfile,"Pk")[0]; //P(k) formula - for heatmap title

const int s=pow(2,log2size);
const int sarr=s*s;
const int nk=ceil(probing*(.5*s*1.415+1)); //number of scales in Pk file



__device__ double krad(int i, int j, int w, int h)
{
	double dist;
	if(i<w/2 and j<h/2) dist=sqrt(i*i+j*j);
	if(i<w/2 and j>=h/2) dist=sqrt(i*i+pow(h-1-j,2));
	if(i>=w/2 and j<h/2) dist=sqrt(pow(w-1-i,2)+j*j);
	if(i>=w/2 and j>=h/2) dist=sqrt(pow(w-1-i,2)+pow(h-1-j,2));
	if(dist<.1){dist=.5;}
	return dist;
}






//random value in [a,b] range - CUDA case
__device__ double Rand(curandState *state, double a, double b) 
{
    double res = a + curand_uniform_double(state) * (b - a);
    return res;
}




enum InterpMethod {linear,lagrange,lin_knownstep};




template <typename T> //interpolating - 1D case
__device__ double Interpolate_CUDA(T *x, T *f, int n, double x0, InterpMethod method)
{
	if(method==linear)
	{
		double x1,x2,y1,y2;

		for(int i=0;i<n-1;++i)
		{
			if(x0>x[i] and x0<=x[i+1])
			{
				x1=x[i];y1=f[i];
				x2=x[i+1];y2=f[i+1];
				break;
			}
		}

		if(x0==x1) return y1;
		else if(x0!=x1) return 1.0*y1+1.0*(((y2-y1)/(x2-x1))*(x0-x1));
		else return f[0];
	}
	
	if(method==lagrange)
	{
		double sum=0.0,add=1.0;

		for(int i=0;i<n-1;++i)
		{   
			add=f[i];
			for(int j=0;j<n-1;++j)
			{
				if(j==i) continue;
				add*=(x0-x[j])/(x[i]-x[j]+1e-10);
			}
			sum+=add;
		}
		return sum;
	}
	
	if(method==lin_knownstep) //regular data, linear interpolation
	{
		if(x0<x[0]){return f[0];}
		if(x0>x[n-1]){return f[n-1];}
		double x1,x2,y1,y2,dx=x[1]-x[0];

		x1=x[0]+floor((x0-x[0])/dx)*dx;
		x2=x1+dx;
		int P1=(x1-x[0])/dx; //locations in arrays
		y1=f[P1];
		y2=f[P1+1];

		if(x0==x1) return y1;
		else if(x0!=x1) return 1.0*y1+1.0*(((y2-y1)/(x2-x1))*(x0-x1));
		else return f[0];
	}
	
	return -1000000;
}






//setting one pixel of Gaussian random field
__global__ void field_onepix(double *field, int s, unsigned long long seed)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x; //[j,k] - th pixel
    int k = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (j >= s || k >= s) return;
	
	double val;
	int loc=k * s + j; //[j,k] pixel -> position on flattened 2D data
	
	curandState state;
	
	curand_init(seed, j,k, &state); // Unique seed per pixel
	val=curand_normal_double(&state); //random drawn from Gauss of sigma=1
	
	field[loc]=val; //setting value
	
	return;
}











//imposing P(k) to one pixel
__global__ void imposePk_onepix(double *field_real, double *field_imag, double *k_listed, double *Pk_listed, int nk, int s)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x; //[j,k] - th pixel
    int k = blockIdx.y * blockDim.y + threadIdx.y;
	
	int loc=k * s + j;
	double kk,val;

	kk=krad(j,k,s,s);
	
	//interpolated
	val=Interpolate_CUDA<double>(k_listed,Pk_listed,nk,kk,lin_knownstep);
	
	//imposing P(k):
	field_real[loc]*=val;
	field_imag[loc]*=val;
	
	
	return;
}




//imposing P(k) on field
void Impose_Pk_CUDA(double *field, double *k_listed, double *Pk_listed)
{
	//fft(dr)****************************
	double *field_imag=new double[sarr]{0.};
	FFT2D_flat(field,field_imag,s,1);
	
	//multiply by Pk*********************
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((s + BLOCK_SIZE - 1) / BLOCK_SIZE, (s + BLOCK_SIZE - 1) / BLOCK_SIZE);
	
	int sarr2 = sarr * sizeof(double);
	int sarr3 = nk * sizeof(double);
	double *d_field,*d_fieldimag,*d_k,*d_Pk; //for data in GPU; field,k,Pk
	
	//allocating memory
	cudaMalloc(&d_field, sarr2); 
	cudaMalloc(&d_fieldimag, sarr2);
	cudaMalloc(&d_k, sarr3);
	cudaMalloc(&d_Pk, sarr3);
	
	//copy data
	cudaMemcpy(d_field,field,sarr2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_fieldimag,field_imag,sarr2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_k,k_listed,sarr3,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pk,Pk_listed,sarr3,cudaMemcpyHostToDevice);
	
	//run on GPU
	imposePk_onepix <<< gridSize, blockSize >>> (d_field,d_fieldimag,d_k,d_Pk,nk,s);
	cudaDeviceSynchronize();
	
	cudaError_t err = cudaGetLastError();
	if (err!=cudaSuccess) //errors which don't show up normally
	{
		cerr <<"CUDA Error: "<<cudaGetErrorString(err)<<endl;
	}
    
	
	//copy data back (k,Pk not necessary anymore)
	cudaMemcpy(field,d_field,sarr2,cudaMemcpyDeviceToHost);
	cudaMemcpy(field_imag,d_fieldimag,sarr2,cudaMemcpyDeviceToHost);
	
	cudaFree(d_field);
	cudaFree(d_fieldimag);
	cudaFree(d_k);
	cudaFree(d_Pk);
	
	//fft^-1******************************
	FFT2D_flat(field,field_imag,s,-1);
	delete []field_imag;
	
	return;
}









void Set_field(double *field, double *k_listed, double *Pk_listed)
{
	//create random Gaussian field**************************************
	double *d_field;
    int sarr2 = sarr* sizeof(double);

    //allocate memory for GPU
    cudaMalloc(&d_field,sarr2);
	
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((s + BLOCK_SIZE - 1) / BLOCK_SIZE, (s + BLOCK_SIZE - 1) / BLOCK_SIZE);
	
	unsigned long long seed = time(NULL)+rand() %12345;
	
	//running each pixel on GPU
	field_onepix <<< gridSize, blockSize >>> (d_field,s,seed);
	cudaDeviceSynchronize();
	
	cudaError_t err = cudaGetLastError();
	if (err!=cudaSuccess) //errors which don't show up normally
	{
		cerr <<"CUDA Error: "<<cudaGetErrorString(err)<<endl;
	}
	
	//copying data back and freeing memory
	cudaMemcpy(field,d_field,sarr2,cudaMemcpyDeviceToHost);
	cudaFree(d_field);

	
	//imposing power spectrum******************************************
	Impose_Pk_CUDA(field,k_listed,Pk_listed);
	cout<<"Gaussian field set, preparing for saving..."<<endl;
	
	return;
}







void Save(double *field)
{
	int loc;
	ofstream f(fout.c_str());
	for(int i=0;i<s;++i)
	{
		if(i%2==0){cout<<setprecision(2)<<"\rSaving: "<<100.*i/s<<"%";}
		for(int j=0;j<s;++j)
		{
			loc=j*s+i;
			f<<i<<" "<<j<<" "<<field[loc]<<endl;
		}
	}
	
	f.close();
	
	return;
}




int main()
{
	srand(time(NULL));
	
	Command("python3 Make_pklist.py"); //making listed P(k)
	double *k_listed=new double[nk];
	double *Pk_listed=new double[nk];
	Fread<double>(pkfile,{k_listed,Pk_listed},{0,1},nk);
	
	double *field=new double[sarr]; //array for storing 2D Gaussian field
	
	Set_field(field,k_listed,Pk_listed); //setting Gaussian field with given P(k)
	Save(field); //saving data
	Command("python3 Plot.py "+fout+" "+fout_img+" "+conv(s)+" '"+Pkform+"'"); //plotting
	
	cout<<endl<<"Output saved in: "<<fout<<", plotted in:"<<fout_img<<", have a nice day:)"<<endl;
	
	return 0;
}
