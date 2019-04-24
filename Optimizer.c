//
//  Optimizer.c
//  GateC
//
//  Created by Steven on 14/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "Optimizer.h"
#include "Link.h"
#include <math.h>
#include  <dlfcn.h>
OptimizerFunc getOptimizer(char* optimizerFuncName)
{
    OptimizerFunc func = dlsym(RTLD_DEFAULT,optimizerFuncName);
    return func;
}

Matrix * getTmpParam(updateParamKey k,int p,Dshape dshape,int datalen)
{
    int key = (int) k + p;
    pNode pnode = findAndCreateLink(0, key, NULL, sizeof(Matrix));
    if(((Matrix*)pnode->data)->array == NULL)
    {
        //printf("ccccc key=%d\n",key);
        Matrix *m = creatMatrixFromValue(0, dshape);
        return m;
    }
    else
    {
        return (Matrix*)(pnode->data);
    }
}

int* getTmpParambyInt(updateParamKey k,int p)
{
    int key = (int) k + p;
    pNode pnode = findAndCreateLink(0, key, NULL, sizeof(Matrix));
    //printf("----------%d key=%d k=%d p=%d\n",*(int*)(pnode->data),key,(int)k,p);
    if((pnode->data) == NULL)
    {
        //printf("hereddddd\n");
        int* data = (int *)MemoryPool_Alloc(sizeof(int));
        //*data = 1;
        memset(data, 1, sizeof(int));
        return data;
    }
    else
    {
        //printf("hereeeee\n");
        return (int*)pnode->data;
    }
}

void SGDOptimizer(pOptimizerParam p)//ok
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

void SGDandMomentumOptimizer(pOptimizerParam p)//ok
{
    double rho = 0.9;
    Matrix *Mw1 = getTmpParam(w1, p->key, p->W->dshape,p->W->length);
    Matrix *Mb1 = getTmpParam(b1, p->key, p->Bias->dshape,p->Bias->length);
    for(int i=0;i<p->dw->length;i++)
    {
        *(Mw1->array + i) = rho * (*(Mw1->array + i)) + (*(p->dw->array + i));
        *(p->W->array + i) -= p->rate * (*(Mw1->array + i));
    }
    for(int i=0;i<p->dbias->length;i++)
    {
        *(Mb1->array + i) = rho * (*(Mb1->array + i)) + (*(p->dbias->array + i));
        *(p->Bias->array + i) -= p->rate * (*(Mb1->array + i));
    }

    findAndCreateLink(1,(int)w1+p->key, Mw1, sizeof(Matrix));
    findAndCreateLink(1,(int)b1+p->key, Mb1, sizeof(Matrix));
}

void SGDNesterovMomentumOptimizer(pOptimizerParam p)//error
{
    double rho = 0.9;
    Matrix *Mw1 = getTmpParam(w1, p->key, p->W->dshape,p->W->length);
    Matrix *Mb1 = getTmpParam(b1, p->key, p->Bias->dshape,p->Bias->length);
    for(int i=0;i<p->dw->length;i++)
    {
        *(Mw1->array + i) = rho * (*(Mw1->array + i)) - p->rate * (*(p->dw->array + i));
        *(Mw1->array + i) = rho * (*(Mw1->array + i)) - p->rate * (*(p->dw->array + i));
        *(p->W->array + i) += *(Mw1->array + i);
    }
    for(int i=0;i<p->dbias->length;i++)
    {
        *(Mb1->array + i) = rho * (*(Mb1->array + i)) - p->rate *(*(p->dbias->array + i));
        *(Mb1->array + i) = rho * (*(Mb1->array + i)) - p->rate *(*(p->dbias->array + i));
        *(p->Bias->array + i) += *(Mb1->array + i);
    }
    
    findAndCreateLink(1,(int)w1+p->key, Mw1, sizeof(Matrix));
    findAndCreateLink(1,(int)b1+p->key, Mb1, sizeof(Matrix));
}

void AdaGradOptimizer(pOptimizerParam p)
{
    
}
void RMSPropOptimizer(pOptimizerParam p)//ok
{
    double decay_rate = 0.99,eps = 1e-10;
    Matrix *Mw1 = getTmpParam(w1, p->key, p->W->dshape,p->W->length);
    Matrix *Mb1 = getTmpParam(b1, p->key, p->Bias->dshape,p->Bias->length);
    for(int i=0;i<p->dw->length;i++)
    {
        *(Mw1->array + i) = decay_rate * (*(Mw1->array + i)) + (1 - decay_rate) * pow(*(p->dw->array + i),2);
        *(p->W->array + i) -= p->rate * (*(p->dw->array + i)) / sqrt(*(Mw1->array + i) + eps);
    }
    for(int i=0;i<p->dbias->length;i++)
    {
        *(Mb1->array + i) = decay_rate * (*(Mb1->array + i)) + (1 - decay_rate) * pow(*(p->dbias->array + i),2);
        *(p->Bias->array + i) -= p->rate * (*(p->dbias->array + i)) / sqrt(*(Mb1->array + i) + eps);
    }
    
    findAndCreateLink(1,(int)w1+p->key, Mw1, sizeof(Matrix));
    findAndCreateLink(1,(int)b1+p->key, Mb1, sizeof(Matrix));
}

void AdamOptimizer(pOptimizerParam p)//ok
{
    double beta1 = 0.9,beta2 = 0.99,eps = 1e-10;
    double mb,vb;
    Matrix *Mw1 = getTmpParam(w1, p->key, p->W->dshape,p->W->length);
    Matrix *Mw2 = getTmpParam(w2, p->key, p->W->dshape,p->W->length);
    Matrix *Mb1 = getTmpParam(b1, p->key, p->Bias->dshape,p->Bias->length);
    Matrix *Mb2 = getTmpParam(b2, p->key, p->Bias->dshape,p->Bias->length);
    int *t = getTmpParambyInt(adam_t,p->key);
    *t = *t + 1;
    //printf("*t = %d\n",*t);
    double p1 = pow(beta1, *t);
    double p2 = pow(beta2,*t);
    for(int i=0;i<p->dw->length;i++)
    {
        *(Mw1->array + i) = beta1 * (*(Mw1->array + i)) + (1 - beta1) * *(p->dw->array + i);
        *(Mw2->array + i) = beta2 * (*(Mw2->array + i)) + (1 - beta2) * pow(*(p->dw->array + i),2);
        mb = *(Mw1->array + i) / (1 - p1);
        vb = *(Mw2->array + i) / (1 - p2);
        *(p->W->array + i) -= p->rate * (mb) / (sqrt(vb) + eps);
    }
    for(int i=0;i<p->dbias->length;i++)
    {
        *(Mb1->array + i) = beta1 * (*(Mb1->array + i)) + (1 - beta1) * *(p->dbias->array + i);
        *(Mb2->array + i) = beta2 * (*(Mb2->array + i)) + (1 - beta2) * pow(*(p->dbias->array + i),2);
        mb = *(Mb1->array + i) / (1 - p1);
        vb = *(Mb2->array + i) / (1 - p2);
        *(p->Bias->array + i) -= p->rate * (mb) / (sqrt(vb) + eps);
    }
    findAndCreateLink(1,(int)w1+p->key, Mw1, sizeof(Matrix));
    findAndCreateLink(1,(int)w2+p->key, Mw2, sizeof(Matrix));
    findAndCreateLink(1,(int)b1+p->key, Mb1, sizeof(Matrix));
    findAndCreateLink(1,(int)b2+p->key, Mb2, sizeof(Matrix));
    findAndCreateLink(1,(int)adam_t+p->key, t, sizeof(int));
}


void AdadeltaOptimizer(pOptimizerParam p)//ok
{
    double rho=0.95,eps=1e-10;
    Matrix *Mw1 = getTmpParam(w1, p->key, p->W->dshape,p->W->length);
    Matrix *Mw2 = getTmpParam(w2, p->key, p->W->dshape,p->W->length);
    Matrix *Mw3 = getTmpParam(w3, p->key, p->W->dshape,p->W->length);
    Matrix *Mb1 = getTmpParam(b1, p->key, p->Bias->dshape,p->Bias->length);
    Matrix *Mb2 = getTmpParam(b2, p->key, p->Bias->dshape,p->Bias->length);
    Matrix *Mb3 = getTmpParam(b3, p->key, p->Bias->dshape,p->Bias->length);
    for(int i=0;i<p->dw->length;i++)
    {
        *(Mw1->array + i) = rho * (*(Mw1->array + i)) + (1 - rho) * pow(*(p->dw->array + i),2);
        *(Mw2->array + i) = sqrt((*(Mw3->array + i) + eps) / (*(Mw1->array + i) + eps)) * (*(p->dw->array + i));
        *(Mw3->array + i) = rho * *(Mw3->array + i) + (1 - rho) * pow(*(Mw2->array + i),2);
        *(p->W->array + i) -= p->rate * *(Mw2->array + i);
    }
    for(int i=0;i<p->dbias->length;i++)
    {
        *(Mb1->array + i) = rho * (*(Mb1->array + i)) + (1 - rho) * pow(*(p->dbias->array + i),2);
        *(Mb2->array + i) = sqrt((*(Mb3->array + i) + eps) / (*(Mb1->array + i) + eps)) * (*(p->dbias->array + i));
        *(Mb3->array + i) = rho * *(Mb3->array + i) + (1 - rho) * pow(*(Mb2->array + i),2);
        *(p->Bias->array + i) -= p->rate * *(Mb2->array + i);
    }
    
    findAndCreateLink(1,(int)w1+p->key, Mw1, sizeof(Matrix));
    findAndCreateLink(1,(int)w2+p->key, Mw2, sizeof(Matrix));
    findAndCreateLink(1,(int)w3+p->key, Mw3, sizeof(Matrix));
    findAndCreateLink(1,(int)b1+p->key, Mb1, sizeof(Matrix));
    findAndCreateLink(1,(int)b2+p->key, Mb2, sizeof(Matrix));
    findAndCreateLink(1,(int)b3+p->key, Mb3, sizeof(Matrix));
}