#include "Incld.h"




void Command(string command) //makes process
{
	FILE *fp;
	int status;
	fp=popen(command.c_str(),"w");
	status=pclose(fp);
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











int FFT(int dir,int m,double *x,double *y)
{
	long nn,i,i1,j,k,i2,l,l1,l2;
	double c1,c2,tx,ty,t1,t2,u1,u2,z;

	/* Calculate the number of points */
	nn = 1;
	for (i=0;i<m;i++)
	{
		nn *= 2;
	}

	/* Do the bit reversal */
	i2 = nn >> 1;
	j = 0;
	for (i=0;i<nn-1;i++)
	{
		if (i < j)
		{
			tx = x[i];
			ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
		}
		k = i2;
		while (k <= j)
		{
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l=0;l<m;l++)
	{
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;
		for (j=0;j<l1;j++)
		{
			for (i=j;i<nn;i+=l2)
			{
				i1 = i + l1;
				t1 = u1 * x[i1] - u2 * y[i1];
				t2 = u1 * y[i1] + u2 * x[i1];
				x[i1] = x[i] - t1;
				y[i1] = y[i] - t2;
				x[i] += t1;
				y[i] += t2;
			}
			z =  u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (dir == 1)
		{
			c2 = -c2;
		}
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for forward transform */
	if (dir == 1)
	{
		for (i=0;i<nn;i++)
		{
			x[i] /= (double)nn;
			y[i] /= (double)nn;
		}
	}


   return true;
}





//2D FFT: (in_real,in_imag)->FFT(in_real,in_imag); real,imag of size [s*s]
int FFT2D_flat(double *img_real, double *img_imag, int s, int direction)
{
   int i,j;
   int m=log2(s);
   int loc_flat;
   double *real,*imag;

   //Transforming rows
   real=(double *)malloc(s * sizeof(double));
   imag=(double *)malloc(s * sizeof(double));
   if(real== NULL or imag==NULL){return false;}

	for (j=0;j<s;++j)
	{
		for (i=0;i<s;++i)
		{
			loc_flat=j * s + i;
			
			real[i]=img_real[loc_flat];
			imag[i]=img_imag[loc_flat];
		}
		FFT(direction,m,real,imag);
		for (i=0;i<s;++i)
		{
			loc_flat=j * s + i;
			
			img_real[loc_flat]=real[i];
			img_imag[loc_flat]=imag[i];
      }
   }
   free(real);
   free(imag);

   //transforming columns
   real=(double *)malloc(s * sizeof(double));
   imag=(double *)malloc(s * sizeof(double));
   if(real==NULL or imag==NULL){return false;}

   for (i=0;i<s;++i)
   {
		for (j=0;j<s;++j)
		{
			loc_flat=j * s + i;
			
			real[j]=img_real[loc_flat];
			imag[j]=img_imag[loc_flat];
      }
      FFT(direction,m,real,imag);
      for (j=0;j<s;++j)
	  {
			loc_flat=j * s + i;
			
			img_real[loc_flat]=real[j];
			img_imag[loc_flat]=imag[j];
      }
   }
   free(real);
   free(imag);

   return true;
}





