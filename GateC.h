//
//  GateC.h
//  GateC
//
//  Created by Steven on 18/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef GateC_h
#define GateC_h
#include <stdio.h>
#include "numpy/ndarrayobject.h"
#include "Matrix.h"
#include "PoolGate.h"
#include "AddGate.h"
#include "CnnGate.h"
#include "NeuronGate.h"
/*
 
 int nd：Numpy Array数组的维度。
 int *dimensions ：Numpy Array 数组每一维度数据的个数。
 int *strides：Numpy Array 数组每一维度的步长。
 char *data： Numpy Array 中指向数据的头指针。

 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
void pyArray2Matrix(PyArrayObject* array,Matrix **m);
PyArrayObject * matrix2pyArray(Matrix* array);
void printArray(PyArrayObject * array);
void printArrayShape(PyArrayObject * array);
#endif /* GateC_h */
