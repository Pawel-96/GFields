Small code generating Gaussian random field with user-provided power spectrum.  
The code computes using GPU with CUDA.

## Requirements
- CUDA-capable GPU with CUDA installed
- cuFFT
- c++14 or newer
- opencv4
- Python 3  
  - numpy, sympy, matplotlib libraries  

## Compilation and running
To compile, type: **make**  
To run, type: **./GField.exe**  

\[Note\]: the code executes external python scripts: Make_pklist.py and Plot.py using system-level calls.

## Parameter file: **param.txt**
- log2size  
&emsp; log_2 of output size, it has to be integer (only 2^n x 2^n output sizes are allowed)  
- Pk  
&emsp; power spectrum formula in python format, e.g. 0.3*(k/100.)**(-1.2)  
- Pk_probing  
&emsp; multiplication factor for number of probed (listed) P(k);  
&emsp; number of datapoints in power spectrum  will be Pk_probing\*kmax (where kmax - maximal k)  
- Pk_precision  
&emsp; precision of P(k) list written into file, e.g. %.6f  
- fname_out  
&emsp; filename of output Gaussian field  
- fname_img  
&emsp; filename of plot with output Gaussian field


## Output
The file **fname_out** defined in param.txt contains Gaussian field saved in ASCII format with columns:  
x, y, f(x,y)  
where x,y are coordinates and f(x,y) is Gaussian field value.  

Additionally, the code creates **fname_img** with visualization of the field and **Pk.txt** with listed P(k) defined by user inside param.txt.


## Examples  
The directory Examples/ contains example runs with Gaussian field (Gaussianfield.txt), plot (Heatmap.png) and power spectrum (Pk.txt) files,  
according to corresponding param.txt parameter file.  


## Troubleshooting
If output contains only zeros, please check the CUDA installation.


## Future improvements
- Implement more robust combination of cpp and python (avoid system calls)

##  
This project uses external libraries including OpenCV and NVIDIA CUDA Toolkit headers (cuFFT, cuRAND).
Users must ensure they have proper installations of these libraries where applicable.
