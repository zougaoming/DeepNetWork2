//
//  CopyGate.c
//  GateC
//
//  Created by Steven on 25/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "CopyGate.h"
void CopyGate_Forward(pCopyGateParam p)
{
    p->_output = p->input;
    if(p->dinput != NULL)
    {
        setZeros(p->dinput);
    }
}

void CopyGate_Backward(pCopyGateParam p)
{
    if(p->dinput == NULL)
    {
        p->dinput = copyMatrix(p->dz);
    }
    else
    {
        addSecondOrderMatrixs2(p->dinput,p->dz);
    }
}