//
//  Gate.c
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "Gate.h"
#include <string.h>
pParamBuff param = NULL;
void printAllParam()
{
    pParamBuff curp = param;
    while(curp != NULL)
    {
        printf("allKey=%d\n",curp->key);
        curp = curp->next;
    }
}
void* insert(pParamBuff paramlist,int key,void *p)
{
    pParamBuff newp = (pParamBuff)MemoryPool_Alloc(sizeof(paramBuff));
    newp->key = key;
    newp->next = NULL;
    newp->p = (PoolGateParam*)MemoryPool_Alloc(sizeof(PoolGateParam));
    memcpy(newp->p,p,sizeof(PoolGateParam));
    //newp->p = p;
    paramlist->next = newp;
    printf("add key=%d\n",key);
    //printarray(((PoolGateParam*)newp->p)->input);
    printAllParam();
    return newp->p;
}
void* findParamBuffAndCreate(int needSave,int key,void *p)
{
    //printf("findKey=%d\n",key);
    printAllParam();
    if(param == NULL){
        param = (pParamBuff)MemoryPool_Alloc(sizeof(paramBuff));
        param->key = key;
        param->p = (PoolGateParam*)MemoryPool_Alloc(sizeof(PoolGateParam));
        memcpy(param->p,p,sizeof(PoolGateParam));
        param->next = NULL;
        return param->p;
    }
    pParamBuff curp = param;
    pParamBuff lastp = param;
    //printf("11111111111 curkey = %d\n",key);
    while(curp != NULL)
    {
        //printf("2222222222 curkey=%d\n",curp->key);
        if(curp->key == key)
        {
            if(needSave == 1){
                memcpy(curp->p,p,sizeof(PoolGateParam));//curp->p = p;
                //curp->p = p;
            }
            //printf("fffffffffffff key=%d\n",key);
            //printarray(((PoolGateParam*)curp->p)->bz_x);
            return curp->p;
        }
        lastp = curp;
        curp = curp->next;
    }
    
    return insert(lastp,key,p);
}





int getOutputSize(int M,int S,int F,int P)
{
    return (int)((M - F + 2 * P) / S + 1);
}



void Optimizer(Matrix *w,Matrix *bias,Matrix *dw,Matrix *db)
{
    OptimizerParam p;
    p.rate = 0.05;
    p.dw = dw;
    p.dbias = db;
    p.W = w;
    p.Bias = bias;
    SGDOptimizer(&p);
}

