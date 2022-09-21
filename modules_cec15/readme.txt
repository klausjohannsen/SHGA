/*
  CEC15 Test Function Suite for Niching Optimization
  Jane Jing Liang 
  email: liangjing@zzu.edu.cn; liangjing@pmail.ntu.edu.cn
  Nov. 21th 2014

  Reference£º	
  B. Y. Qu, J. J. Liang, P. N. Suganthan, Q. Chen, "Problem Definitions and Evaluation Criteria for the CEC 2015 Special Session and Competition on Niching Numerical Optimization",Technical Report201411B,Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, November 2014
*/

cec15_nich_func.cpp is the test function
Example:
cec15_nich_func(x, f, dimension,population_size,func_num);

main.cpp is an example function about how to use cec15_test_func.cpp


#include <WINDOWS.H>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>
void cec15_nich_func(double *, double *,int,int,int);
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag;
void main()
{
...
}

For Linux Users:
Please  change %xx in fscanf and fprintf and do use "WINDOWS.H". 

