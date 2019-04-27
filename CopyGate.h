//
//  CopyGate.h
//  GateC
//
//  Created by Steven on 25/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef CopyGate_h
#define CopyGate_h
#include "Matrix.h"
#include "Gate.h"
#include <stdio.h>
typedef struct CopyGateParamS{
    Matrix *input;
    Matrix *_output;
    Matrix *dz;
    Matrix *dinput;
    int key;
}CopyGateParam,*pCopyGateParam;
void CopyGate_Forward(pCopyGateParam p);
void CopyGate_Backward(pCopyGateParam p);
#endif /* CopyGate_h */
