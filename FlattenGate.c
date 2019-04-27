//
//  FlattenGate.c
//  GateC
//
//  Created by Steven on 26/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "FlattenGate.h"
void FlattenGate_Forward(pFlattenGateParam p)
{
    int N = p->input->dshape.shape[0];
    int L = p->input->dshape.shape[1] * p->input->dshape.shape[2] * p->input->dshape.shape[3];
    if(p->_output == NULL)
    {
        Dshape newshape;
        initDshapeInt(&newshape, 0, 0, L,N);
        p->_output = creatZerosMatrix(newshape);
    }
    int x,y,z;
    z = p->input->dshape.shape[3];
    y = p->input->dshape.shape[2] * z;
    x = p->input->dshape.shape[1] * y;
    int idx = 0;
    for(int n =0;n<p->input->dshape.shape[0];n++)
    {
        idx = 0;
        for(int c =0;c<p->input->dshape.shape[1];c++)
        {
            for(int h =0;h<p->input->dshape.shape[2];h++)
            {
                for(int w =0;w<p->input->dshape.shape[3];w++)
                {
                    *(p->_output->array + n + idx * N ) = *(p->input->array + n * x + c * y + h * z + w);
                    idx ++;
                }
            }
        }
    }
}
void FlattenGate_Backward(pFlattenGateParam p)
{
    int N = p->_output->dshape.shape[3];
    if(p->dinput == NULL)
    {
        p->dinput = zeros_like(p->input);
    }
    else{
        setZeros(p->dinput);
    }
   
    int x,y,z;
    z = p->dinput->dshape.shape[3];
    y = p->dinput->dshape.shape[2] * z;
    x = p->dinput->dshape.shape[1] * y;
    int idx = 0;
    for(int n =0;n<p->input->dshape.shape[0];n++)
    {
        idx = 0;
        for(int c =0;c<p->input->dshape.shape[1];c++)
        {
            for(int h =0;h<p->input->dshape.shape[2];h++)
            {
                for(int w =0;w<p->input->dshape.shape[3];w++)
                {
                    *(p->dinput->array + n * x + c * y + h * z + w) = *(p->dz->array + n + idx * N);
                    idx ++;
                }
            }
        }
    }
}