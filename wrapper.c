//
//  wrapper.c
//  GateC
//
//  Created by Steven on 18/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "wrapper.h"
#include <Python.h>
#include <stdlib.h>
#include "GateC.h"
#include "Gate.h"
#include <string.h>
#include "Eval.h"
/*
typedef struct  {
    int row, col;
    float *element;
    unsigned char init;
}Mat;
*/
int str_len(char *s)
{
    int len=0;
    while(*s!='\0')
    {
        len++;
        s++;
    }
    return len;
}

static PyObject *py_PoolGate_backward(PyObject *self,PyObject * args)
{
    PyObject* p_dz;
    int key;
    if(!PyArg_ParseTuple(args,"Oi",&p_dz,&key))
    {
        return NULL;
    }
    pNode node = findAndCreateLink(0,key,NULL,sizeof(PoolGateParam));
    PoolGateParam* newp = (PoolGateParam*)(node->data);
    
    //findParamBuffAndCreate(0,key,&p);
    //printarray(newp->bz_x);
    //printarray(pyArray2Matrix((PyArrayObject*)bz_x));
    //newp->strids = p.strids;
    //newp->type = p.type;
    //newp->input = pyArray2Matrix((PyArrayObject*)p_input);
    newp->dz = pyArray2Matrix((PyArrayObject*)p_dz);
    //newp->bz_x = pyArray2Matrix((PyArrayObject*)bz_x);
    //newp->bz_y = pyArray2Matrix((PyArrayObject*)bz_y);
    
    PyArrayObject *result =  matrix2pyArray(PoolGate_Backward(newp));
    
    //destroyMatrix(newp->input);
    //destroyMatrix(newp->dz);
    //destroyMatrix(newp->bz_x);
    //destroyMatrix(newp->bz_y);
    return (PyObject *)Py_BuildValue("O",result);
}
static PyObject *py_PoolGate_init(PyObject *self,PyObject * args)
{
    PyObject* p_object;
    PoolGateParam p;
    int key;
    if(!PyArg_ParseTuple(args,"Oiisi",&p_object,&(p.filter),&(p.strids),&(p.type),&key))
    {
        return NULL;
    }
    p.input = pyArray2Matrix((PyArrayObject*)p_object);
    pNode node = findAndCreateLink(0,key,&p,sizeof(PoolGateParam));
    if(node == NULL)
        return (PyObject *)Py_BuildValue("i",0);
    else
        return (PyObject *)Py_BuildValue("i",1);
}

static PyObject *py_PoolGate(PyObject *self,PyObject * args)
{
    PyObject* p_object;
    PoolGateParam p;
    int key;
    if(!PyArg_ParseTuple(args,"Oiisi",&p_object,&(p.filter),&(p.strids),&(p.type),&key))
    {
        return NULL;
    }
    p.input = pyArray2Matrix((PyArrayObject*)p_object);
    PoolGateParam *curp = &p;
    PyArrayObject *result =  matrix2pyArray(PoolGate_Forward(curp));
    //destroyMatrix(p.input);
    findAndCreateLink(1,key,curp,sizeof(PoolGateParam));
    if(strcmp(p.type, "MAX") == 0)
    {
        PyArrayObject* bz_x = matrix2pyArray(curp->bz_x);
        PyArrayObject* bz_y = matrix2pyArray(curp->bz_y);
        return (PyObject *)Py_BuildValue("OOO",result,bz_x,bz_y);
    }
    
    return (PyObject *)Py_BuildValue("O",result);
}
static PyObject *py_CnnGate_backward(PyObject *self,PyObject * args)
{
    PyObject *p_dz;
    //CnnGateParam p;
    int key;
    char* activefunc;
    if(!PyArg_ParseTuple(args,"Ois",&p_dz,&key,&activefunc))
    {
        return NULL;
    }
    CnnGateParam* newp = (CnnGateParam*)(findAndCreateLink(0,key,NULL,sizeof(CnnGateParam))->data);
    
    newp->backward = getBackward(activefunc);
    //printf("key=%d,s=%s\n",key,newp->activefunc);
    //p.input = pyArray2Matrix((PyArrayObject*)p_input);
    //p._output = pyArray2Matrix((PyArrayObject*)p_output);
    //p.weight = pyArray2Matrix((PyArrayObject*)p_weight);
    //p.bias = pyArray2Matrix((PyArrayObject*)p_bias);
    newp->dz = pyArray2Matrix((PyArrayObject*)p_dz);
    Backward(newp);
    PyArrayObject *dw =  matrix2pyArray(newp->dw);
    PyArrayObject *dx =  matrix2pyArray(newp->dx);
    PyArrayObject *dbias =  matrix2pyArray(newp->dbias);
    
    destroyMatrix(newp->input);
    destroyMatrix(newp->_output);
    destroyMatrix(newp->weight);
    destroyMatrix(newp->bias);
    destroyMatrix(newp->dz);
    return (PyObject *)Py_BuildValue("OOO",dw,dbias,dx);
    
}

static PyObject *py_CnnGate(PyObject *self,PyObject * args)
{
    
    PyObject* p_object;
    PyObject* p_weight;
    PyObject* p_bias;
    CnnGateParam p;
    CnnGateParam *curp = &p;
    char* activefunc;
    int key;
    if(!PyArg_ParseTuple(args,"O|O|Osiii",&p_object,&(p_weight),&(p_bias),&(activefunc),&(curp->strids),&(curp->panding),&key))
    {
        return NULL;
    }

    curp->forward = getForward(activefunc);
    //printf("forward key=%d,s=%s\n",key,activefunc);
    curp->backward = getBackward(activefunc);
    //curp->isBackward = p.isBackward;
    curp->weight = pyArray2Matrix((PyArrayObject*)p_weight);
    curp->bias = pyArray2Matrix((PyArrayObject*)p_bias);
    curp->input = pyArray2Matrix((PyArrayObject*)p_object);
    Forward(curp);
    findAndCreateLink(1,key,curp,sizeof(CnnGateParam));
    PyArrayObject *result =  matrix2pyArray(curp->_output);
    PyObject *  t = (PyObject *)Py_BuildValue("O",result);
    //destroyMatrix(curp->weight);
    //destroyMatrix(curp->bias);
    //destroyMatrix(curp->input);
    return t;
}

static PyObject *py_NeuronGate_backward(PyObject *self,PyObject * args)
{
    
    PyObject* p_input;
    PyObject* p_output;
    PyObject *p_dz;
    PyObject* p_weight;
    PyObject* p_bias;
    PyObject* p_activefunc;
    NeuronGateParam p;
    if(!PyArg_ParseTuple(args,"O|O|O|O|O|s",&p_input,&p_output,&p_dz,&(p_weight),&(p_bias),&p_activefunc))
    {
        return NULL;
    }
    p.backward = getBackward((char*)p_activefunc);
    p.input = pyArray2Matrix((PyArrayObject*)p_input);
    p._output = pyArray2Matrix((PyArrayObject*)p_output);
    p.weight = pyArray2Matrix((PyArrayObject*)p_weight);
    p.bias = pyArray2Matrix((PyArrayObject*)p_bias);
    p.dz = pyArray2Matrix((PyArrayObject*)p_dz);
    Neuron_Backward(&p);
    PyArrayObject *dw =  matrix2pyArray(p.dw);
    PyArrayObject *dx =  matrix2pyArray(p.dx);
    PyArrayObject *dbias =  matrix2pyArray(p.dbias);
    
    destroyMatrix(p.input);
    destroyMatrix(p._output);
    destroyMatrix(p.weight);
    destroyMatrix(p.bias);
    destroyMatrix(p.dz);
    return (PyObject *)Py_BuildValue("OOO",dw,dbias,dx);
    
}

static PyObject *py_NeuronGate_Forward(PyObject *self,PyObject * args)
{
    
    PyObject* p_input;
    PyObject* p_weight;
    PyObject* p_bias;
    PyObject* p_activefunc;
    NeuronGateParam p;
    if(!PyArg_ParseTuple(args,"O|O|O|s",&p_input,&(p_weight),&(p_bias),&p_activefunc))
    {
        return NULL;
    }
    
    NeuronGateParam *curp = &p;
    curp->forward = getForward((char*)p_activefunc);
    curp->weight = pyArray2Matrix((PyArrayObject*)p_weight);
    curp->bias = pyArray2Matrix((PyArrayObject*)p_bias);
    curp->input = pyArray2Matrix((PyArrayObject*)p_input);
    Neuron_Forward(curp);
    PyArrayObject *result =  matrix2pyArray(curp->_output);
    PyObject *  t = (PyObject *)Py_BuildValue("O",result);
    destroyMatrix(curp->weight);
    destroyMatrix(curp->bias);
    destroyMatrix(curp->input);
    return t;
}
static PyObject *py_test(PyObject *self,PyObject * args)
{
    
    PyObject* p_test;
    PyObject* p_test2;
    if(!PyArg_ParseTuple(args,"O|O",&p_test,&p_test2))
    {
        return NULL;
    }
    Matrix * test = pyArray2Matrix((PyArrayObject *)p_test);
    Matrix * test2 = pyArray2Matrix((PyArrayObject *)p_test2);
    
    test_eval();
    SoftmaxActivator_Backward(test,test2);
    PyArrayObject *result =  matrix2pyArray(test);
    
    printf("%f\n",get_mempool_usage());
    destroyMatrix(test);
    destroyMatrix(test2);
    printf("%f\n",get_mempool_usage());
    return (PyObject *)Py_BuildValue("O",result);
}

static PyObject *py_AddGate_Forward(PyObject *self,PyObject * args)
{
    PyObject *obj1;
    PyObject *obj2;
    AddGateParam p;
    if(!PyArg_ParseTuple(args,"O|O",&obj1,&obj2))
    {
        return NULL;
    }
    AddGateParam* curp = &p;
    curp->input1 = pyArray2Matrix((PyArrayObject*)obj1);
    curp->input2 = pyArray2Matrix((PyArrayObject*)obj2);
    AddGate_Forward(curp);
    PyArrayObject *result = matrix2pyArray(curp->_output);
    return (PyObject *)Py_BuildValue("O",result);
}

static PyObject *py_AddGate_Backward(PyObject *self,PyObject * args)
{
    
    PyObject *obj1;
    AddGateParam p;
    if(!PyArg_ParseTuple(args,"O",&obj1))
    {
        return NULL;
    }
    
    AddGateParam* curp = &p;
    curp->dz = pyArray2Matrix((PyArrayObject*)obj1);
    AddGate_Backward(curp);
    PyArrayObject *result = matrix2pyArray(curp->_output);
    return (PyObject *)Py_BuildValue("O",result);
}
//添加模块数组(注意是PyMethodDef,不要错写成PyMethondDef)
//定义对应的方法名，后面Python调用的时候就用这里面的方法名调用
static PyMethodDef py_test1Methods[] = {
    {"PoolGate_init",py_PoolGate_init,METH_VARARGS},
    {"PoolGate",py_PoolGate,METH_VARARGS},
    {"PoolGate_Backward",py_PoolGate_backward,METH_VARARGS},
    {"AddGate_Forward",py_AddGate_Forward,METH_VARARGS},
    {"AddGate_Backward",py_AddGate_Backward,METH_VARARGS},
    {"CnnGate",py_CnnGate,METH_VARARGS},
    {"CnnGate_Backward",py_CnnGate_backward,METH_VARARGS},
    {"NeuronGate_Forward",py_NeuronGate_Forward,METH_VARARGS},
    {"NeuronGate_Backward",py_NeuronGate_backward,METH_VARARGS},
    {"test",py_test,METH_VARARGS},
    
    
    {NULL,NULL},
};
static struct PyModuleDef ExtestModule =
{
    PyModuleDef_HEAD_INIT,
    "GateC", NULL, -1, py_test1Methods
};
void init_numpy2()
{
    Py_Initialize();
    {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");  } }

}
PyMODINIT_FUNC PyInit_GateC(void)
{
    //printf("this");
    getMemoryPool();
    Py_Initialize();
    if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); }
    return PyModule_Create(&ExtestModule);
}
