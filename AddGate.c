//
//  AddGate.c
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "AddGate.h"
void AddGate_Forward(AddGateParam *p)
{
    p->_output = addSecondOrderMatrixs(p->input1,p->input2);
}
void AddGate_Backward(AddGateParam *p)
{
    p->_output = p->dz;
}
