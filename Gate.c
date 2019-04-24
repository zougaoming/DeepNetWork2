//
//  Gate.c
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "Gate.h"
#include <string.h>

int getOutputSize(int M,int S,int F,int P)
{
    return (int)((M - F + 2 * P) / S + 1);
}



void Optimizer(Matrix *w,Matrix *bias,Matrix *dw,Matrix *db,int key,char* optimizerFuncName)
{
    OptimizerParam p;
    p.rate = 0.05;
    p.dw = dw;
    p.dbias = db;
    p.W = w;
    p.Bias = bias;
    p.key = key;
    OptimizerFunc optimizerFunc = getOptimizer(optimizerFuncName);
    if(optimizerFunc == NULL){
        printf("Optimizer暂时不支持%s,继续默认调用函数\"SGDOptimizer()\"!\n",optimizerFuncName);
        return SGDOptimizer(&p);
    }
    optimizerFunc(&p);
}

