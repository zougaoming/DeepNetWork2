//
//  Gate.h
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef Gate_h
#define Gate_h

#include <stdio.h>
#include "Activatior.h"
#include "Optimizer.h"
#include "Link.h"

int findParamBuff(int key);

int getOutputSize(int M,int S,int F,int P);


void Optimizer(Matrix *w,Matrix *bias,Matrix *dw,Matrix *db);
#endif /* Gate_h */
