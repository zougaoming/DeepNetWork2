//
//  NeuronGate.c
//  GateC
//
//  Created by Steven on 1/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "NeuronGate.h"
void Neuron_Forward(NeuronGateParam *p){
    transposeSecondOrderMatrix(p->weight);
    p->_output = mulSecondOrderMatrixs(p->weight,p->input);
    for(int i = 0;i< p->_output->dshape.shape[2];i++)
    {
        for(int j = 0;j< p->_output->dshape.shape[3];j++)
        {
            *(p->_output->array + i * p->_output->dshape.shape[3] + j) += *(p->bias->array + i * p->bias->dshape.shape[3]);
        }
    }
    
    p->forward(p->_output);
    
}
void Neuron_Backward(NeuronGateParam *p){
    if(p->dbias == NULL) p->dbias = zeros_like(p->bias);
    p->backward(p->dz,p->_output);
    transposeSecondOrderMatrix(p->input);
    p->dw = mulSecondOrderMatrixs(p->dz,p->input);
    transposeSecondOrderMatrix(p->dw);
    transposeSecondOrderMatrix(p->input);//转回来
    
    transposeSecondOrderMatrix(p->weight);
    p->dx = mulSecondOrderMatrixs(p->weight,p->dz);
    //transposeSecondOrderMatrix(p->weight);
    double sum = 0;
    for(int i = 0;i<p->dz->dshape.shape[2];i++)
    {
        sum = 0;
        for(int j = 0;j<p->dz->dshape.shape[3];j++)
        {
            sum += *(p->dz->array + i * p->dz->dshape.shape[3] + j);
        }
        *(p->dbias->array + i * p->dbias->dshape.shape[3]) = sum;
    }
    Optimizer(p->weight,p->bias,p->dw,p->dbias,p->key,p->optimizerFuncName);
}