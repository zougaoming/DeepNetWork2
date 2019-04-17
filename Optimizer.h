//
//  Optimizer.h
//  GateC
//
//  Created by Steven on 14/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef Optimizer_h
#define Optimizer_h

#include <stdio.h>
#include "Matrix.h"
typedef struct OptimizerParams{
    double rate;
    Matrix *dw;
    Matrix *dbias;
    Matrix *W;
    Matrix *Bias;
    Matrix *W1;
    Matrix *W2;
    Matrix *W3;
    Matrix *b1;
    Matrix *b2;
    Matrix *b3;
    unsigned int t;
}*pOptimizerParam,OptimizerParam;

void SGDOptimizer(pOptimizerParam p);
#endif /* Optimizer_h */
