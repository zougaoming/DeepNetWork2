//
//  FlattenGate.h
//  GateC
//
//  Created by Steven on 26/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef FlattenGate_h
#define FlattenGate_h
#include "Matrix.h"
#include "Gate.h"
#include <stdio.h>
typedef struct FlattenGateParamS{
    Matrix *input;
    Matrix *_output;
    Matrix *dz;
    Matrix *dinput;
    int key;
}FlattenGateParam,*pFlattenGateParam;
void FlattenGate_Forward(pFlattenGateParam p);
void FlattenGate_Backward(pFlattenGateParam p);

#endif /* FlattenGate_h */
