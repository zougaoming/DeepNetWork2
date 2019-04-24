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
    unsigned int tmpKey;
    int key;
}*pOptimizerParam,OptimizerParam;

typedef void (*OptimizerFunc)(pOptimizerParam p);
OptimizerFunc getOptimizer(char* optimizerFuncName);

void SGDOptimizer(pOptimizerParam p);
void SGDandMomentumOptimizer(pOptimizerParam p);
void SGDNesterovMomentumOptimizer(pOptimizerParam p);
void AdaGradOptimizer(pOptimizerParam p);
void RMSPropOptimizer(pOptimizerParam p);
void AdamOptimizer(pOptimizerParam p);
void AdadeltaOptimizer(pOptimizerParam p);
#endif /* Optimizer_h */
