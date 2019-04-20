//
//  Optimizer.c
//  GateC
//
//  Created by Steven on 14/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "Optimizer.h"
void SGDOptimizer(pOptimizerParam p)
{
    if(p->rate == 0)p->rate = 0.05;
    for(int i=0;i<p->dw->length;i++)
    {
        *(p->W->array + i) -= p->rate * *(p->dw->array + i);
    }
    for(int i=0;i<p->dbias->length;i++)
    {
        *(p->Bias->array + i) -= p->rate * *(p->dbias->array + i);
    }
}


