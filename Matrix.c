//
//  Matrix.c
//  GateC
//
//  Created by Steven on 19/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <sys/malloc.h>
#include <math.h>
#include "Matrix.h"
#include "memorypool.h"

//从连续的数据创建数组,浅拷贝
Matrix *creatAsMatrixFromDatas(double *data,int data_len, Dshape dshape){
    if(!data) return NULL;
    Matrix *m = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!m) return NULL;
    m->array = data;
    m->dshape = dshape;
    m->length = data_len;
    m->size = data_len;
    return m;
}
void updateMatrixData()
{
    
}
//从连续的数据创建数组,深拷贝
Matrix *creatMatrixFromDatas(double *data,int data_len, Dshape dshape){
    int i = 0;
    if(!data) return NULL;
    Matrix *m = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!m) return NULL;
    m->array = (double *)MemoryPool_Alloc(data_len*sizeof(double));
    if(!m->array){
        free(m);
        return NULL;
    }
    for(i=0;i<data_len;i++){
        *(m->array+i) = *(data+i);
    }
    m->dshape = dshape;
    m->length = data_len;
    m->size = data_len;
    return m;
}

//创建单值数组
Matrix *creatMatrixFromValue(double value, Dshape dshape){
    int i = 0,w,x,y,z,data_len=0;
    z = dshape.shape[3];
    y = dshape.shape[2] * z;
    x = dshape.shape[1] * y;
    w = dshape.shape[0] * x;
    if(w != 0){
        data_len = w;
    }else if(x != 0){
        data_len = x;
    }else if(y != 0){
        data_len = y;
    }else if(z != 0){
        data_len = z;
    }
    Matrix *m = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!m) return NULL;
    m->array = (double *)MemoryPool_Alloc(data_len*sizeof(double));
    if(!m->array){
        MemoryPool_Free(m);
        return NULL;
    }
    for(i=0;i<data_len;i++){
        *(m->array+i) = value;
    }
    m->dshape = dshape;
    m->length = data_len;
    m->size = data_len;
    return m;
}

//指定步长，创建等间隔值的数组
Matrix *creatMatrixFromArange(double startVal, double stepVal,Dshape dshape){
    int i = 0,w,x,y,z,data_len=0;
    z = dshape.shape[3];
    y = dshape.shape[2] * z;
    x = dshape.shape[1] * y;
    w = dshape.shape[0] * x;
    if(w != 0){
        data_len = w;
    }else if(x != 0){
        data_len = x;
    }else if(y != 0){
        data_len = y;
    }else if(z != 0){
        data_len = z;
    }
    Matrix *m = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!m) return NULL;
    m->array = (double *)MemoryPool_Alloc(data_len*sizeof(double));
    if(!m->array){
        free(m);
        return NULL;
    }
    for(i=0;i<data_len;i++){
        *(m->array+i) = startVal + stepVal * i;
    }
    m->dshape = dshape;
    m->length = data_len;
    m->size = data_len;
    return m;
}

//指定范围之间的均匀间隔数量，创建等间隔值的数组
Matrix *creatMatrixFromLinspace(double startVal, double endVal,Dshape dshape){
    int i = 0,w,x,y,z,data_len=0;
    double stepVal;
    z = dshape.shape[3];
    y = dshape.shape[2] * z;
    x = dshape.shape[1] * y;
    w = dshape.shape[0] * x;
    if(w != 0){
        data_len = w;
    }else if(x != 0){
        data_len = x;
    }else if(y != 0){
        data_len = y;
    }else if(z != 0){
        data_len = z;
    }
    Matrix *m = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!m) return NULL;
    m->array = (double *)MemoryPool_Alloc(data_len*sizeof(double));
    if(!m->array){
        free(m);
        return NULL;
    }
    stepVal = (endVal - startVal) / data_len;
    for(i=0;i<data_len;i++){
        *(m->array+i) = startVal + stepVal * i;
    }
    m->dshape = dshape;
    m->length = data_len;
    m->size = data_len;
    return m;
}

Matrix * zeros_like(Matrix *m)
{
    return creatZerosMatrix(m->dshape);
}

Matrix * ones_like(Matrix *m)
{
    return creatOnesMatrix(m->dshape);
}

//创建全0数组
Matrix *creatZerosMatrix(Dshape dshape){
    return creatMatrixFromValue(0, dshape);
}

//创建全1数组
Matrix *creatOnesMatrix(Dshape dshape){
    return creatMatrixFromValue(1, dshape);
}
//置为0
int setZeros(Matrix *m)
{
    if(!m)return 0;
    for(int i=0;i<m->length;i++)
    {
        *(m->array + i) = 0;
    }
    return 1;
}


//创建二维单位阵
Matrix *creatIdentitySecondOrderMatrix(Dshape dshape){
    if(dshape.shape[0] != 0 || dshape.shape[1] != 0 ||
       dshape.shape[2] <= 1 || dshape.shape[2] != dshape.shape[3]) return NULL;//dshape不是二维方阵
    int i = 0,data_len;
    data_len = dshape.shape[2] * dshape.shape[3];
    Matrix *m = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!m) return NULL;
    m->array = (double *)MemoryPool_Alloc(data_len*sizeof(double));
    if(!m->array){
        MemoryPool_Free(m);
        return NULL;
    }
    for(i=0;i<data_len;i++){
        *(m->array+i) = 0;
    }
    for(i=0;i<dshape.shape[3];i++){
        *(m->array+i*dshape.shape[3] + i) = 1;
    }
    m->dshape = dshape;
    m->length = data_len;
    m->size = data_len;
    return m;
}

//数组的shape初始化
void initDshape(Dshape *dshape,int shapeval[]){
    if(!dshape) return;
    for(int i = 0;i < 4;i++)
    {
        dshape->shape[i] = shapeval[i];
    }
}
//数组的shape初始化
void initDshapeInt(Dshape *dshape,int s_1,int s_2,int s_3,int s_4){
    if(!dshape) return;
    if(s_1 != -1) dshape->shape[0] =s_1;
    if(s_2 != -1) dshape->shape[1] =s_2;
    if(s_3 != -1) dshape->shape[2] =s_3;
    if(s_4 != -1) dshape->shape[3] =s_4;
}

//清空数组，保留空间
void clearMatrix(Matrix *m){
    if(!m) return;
    m->length = 0;
}

//删除数组，并释放空间
void destroyMatrix(Matrix *m){
    if(m){
        if(m->array){
            MemoryPool_Free(m->array);
            m->array = NULL;
        }
        MemoryPool_Free(m);
    }
    m = NULL;
}

//重新调整数组shape
int reshape(Matrix *m,Dshape dshape){
    if(!m) return -1;
    m->dshape = dshape;
    return 0;
}

//打印数组shape
void printShape(Matrix *m){
    int i = 0,j;
    if(!m) return;
    printf("(");
    while(m->dshape.shape[i] == 0){
        i++;
    }
    printf("%d",m->dshape.shape[i]);
    i++;
    for(j=i;j<4;j++){
        printf(", %d",m->dshape.shape[j]);
    }
    printf(")\n");
}

//获取维度的长度
int getIndexMaterx(Matrix *m,int dim)
{
    switch (dim) {
        case 3:
            return m->dshape.shape[3];
            break;
        case 2:
            return m->dshape.shape[2] * m->dshape.shape[3];
            break;
        case 1:
            return m->dshape.shape[1] * m->dshape.shape[2] * m->dshape.shape[3];
            break;
        case 0:
            return  m->dshape.shape[0] * m->dshape.shape[1] * m->dshape.shape[2] * m->dshape.shape[3];
            break;
        default:
            return 0;
            break;
    }
}
//获取指定下表的指针偏差
int getDiffPointerMaterx(Matrix *m,int dim0,int dim1,int dim2,int dim3)
{
    return dim0 * m->dshape.shape[1] * m->dshape.shape[2] * m->dshape.shape[3] + dim1* m->dshape.shape[2] * m->dshape.shape[3] + dim2* m->dshape.shape[3] + dim3;
}

//获得数组的维数，最多4维
int getMatrixNdim(Matrix *m){
    int i,ndim = 0;
    for(i=0;i<4;i++){
        if(m->dshape.shape[i] != 0){
            ndim ++;
        }
    }
    return ndim;
}

//打印数组
void printarray(Matrix *m){
    int i,w,x,y,z;
    if(!m){
        printf("ERROR printarray m is NULL!!!\n");
        return;
    }
    z = m->dshape.shape[3];
    y = m->dshape.shape[2] * z;
    x = m->dshape.shape[1] * y;
    w = m->dshape.shape[0] * x;
    //printf("%d",m->length);
    if(m->length > 0){
        for(i=0;i<3;i++){
            if(m->dshape.shape[i] != 0){
                printf("[");
            }
        }
        for(i=0;i<m->length;i++){
            if(i%z == 0){
                if(i == 0){
                    printf("[");
                }else{
                    if(x != 0){
                        if(i%x == 0){
                            printf("]]]\n[[[");
                        }else if(i%y == 0){
                            printf("]]\n[[");
                        }else{
                            printf("], [");
                        }
                    }else if(y != 0){
                        if(i%y == 0){
                            printf("]]\n[[");
                        }else{
                            printf("]\n[");
                        }
                    }
                }
            }else{
                printf(", ");
            }
            printf("%g",*(m->array + i));
        }
        for(i=0;i<4;i++){
            if(m->dshape.shape[i] != 0){
                printf("]");
            }
        }
    }
    printf("\n");
}
//获取二维数组，最后两维
void get2dim(Matrix *m,Matrix*mto,int dimen0,int dimen1)
{
    
    if(!m) return ;
    int w,x,y,z;
    z = m->dshape.shape[3];
    y = m->dshape.shape[2] * z;
    x = m->dshape.shape[1] * y;
    w = m->dshape.shape[0] * x;
    int strati = dimen0 *x + dimen1 * y;
    for(int i=0;i<mto->dshape.shape[2];i++){
        for(int j=0;j<mto->dshape.shape[3];j++){
            //printf("fromi=%d fromlen=%d        toi=%d tolen=%d\n",strati + i*z+j,m->length,i*z+j,mto->length);
            *(mto->array+i*z+j) = *(m->array+ strati + i*z+j);
            
        }
    }
    //return subm;
    
}

//求二维方阵数组的迹
double getSecondOrderMatrixTrace(Matrix *m){
    double result = 0;
    int i;
    if(!m) return -1;
    if(m->dshape.shape[0] != 0 || m->dshape.shape[1] != 0 ||
       m->dshape.shape[2] == 0) return 0; //非二阶矩阵
    if(m->dshape.shape[2] != m->dshape.shape[3]) return 0; //非方阵
    for(i=0;i<m->dshape.shape[2];i++){
        result += *(m->array + i*m->dshape.shape[2] + i);
    }
    return result;
}

//在二维数组m1的行末尾按行拼接上m2，要求行的长度要一致
int spliceSecondOrderMatrixRow(Matrix *m1,Matrix *m2){
    int i;
    if(!m1 || !m2) return -1;
    if(m1->dshape.shape[3] != m2->dshape.shape[3]) return -1;
    m1->array = (double *)realloc(m1->array,(m1->length+m2->length)*sizeof(double));
    if(!m1->array) return -1;
    for(i=0;i<m2->length;i++){
        *(m1->array + m1->length + i) = *(m2->array + i);
    }
    if(m1->dshape.shape[2] == 0){
        if(m2->dshape.shape[2] == 0){
            m1->dshape.shape[2] = 2;
        }else{
            m1->dshape.shape[2] = 1 + m2->dshape.shape[2];
        }
    }else{
        if(m2->dshape.shape[2] == 0){
            m1->dshape.shape[2] += 1;
        }else{
            m1->dshape.shape[2] += m2->dshape.shape[2];
        }
    }
    m1->size = m1->length+m2->length;
    m1->length += m2->length;
    return 0;
}

//在二维数组m1的列末尾按列拼接上m2，要求列的长度要一致
int spliceSecondOrderMatrixColume(Matrix *m1,Matrix *m2){
    if(transposeSecondOrderMatrix(m1) == -1) return -1;
    if(transposeSecondOrderMatrix(m2) == -1) return -1;
    if(spliceSecondOrderMatrixRow(m1,m2) == -1)return -1;
    if(transposeSecondOrderMatrix(m1) == -1) return -1;
    if(transposeSecondOrderMatrix(m2) == -1) return -1;
    return 0;
}
//在M1的最后两维度，前面和后面，上面和下面都增加一个zeropanding
int PandingMatrix4D(Matrix *m,unsigned zeropanding)
{
    if(zeropanding <=0 || !m) return 0;
    int z,y,x,w;
    int z1,y1,x1,w1;
    Matrix * m2 = copyMatrix(m);
    m->dshape.shape[2] += 2*zeropanding;
    m->dshape.shape[3] += 2*zeropanding;
    int newLength = m->dshape.shape[0] * m->dshape.shape[1] * m->dshape.shape[2] * m->dshape.shape[3];
    m->length = newLength;
    m->size = newLength;
    
    z = m->dshape.shape[3];
    y = m->dshape.shape[2] * z;
    x = m->dshape.shape[1] * y;
    w = m->dshape.shape[0] * x;
    z1 = m2->dshape.shape[3];
    y1 = m2->dshape.shape[2] * z1;
    x1 = m2->dshape.shape[1] * y1;
    w1 = m2->dshape.shape[0] * x1;
    MemoryPool_Free(m->array);
    m->array = (double *)MemoryPool_Alloc(newLength * sizeof(double));
    //m->array = (double *)MemoryPool_realloc(m->array,(m->length)*sizeof(double));
    
    for(int i = 0;i<m->length;i++)
    {
        *(m->array + i) = .0;
    }

    for(int l1 = 0;l1<m2->dshape.shape[0];l1++)
    {
        for(int l2 = 0;l2<m2->dshape.shape[1];l2++)
        {
            for(int i = 0;i<m2->dshape.shape[2];i++)
            {
                for(int j = 0;j<m2->dshape.shape[3];j++)
                {
                    *(m->array + l1*x + l2*y + (i + zeropanding) * z + (j + zeropanding) ) = *(m2->array + l1 * x1 + l2*y1 + i * m2->dshape.shape[3] + j);
                }
            }

        }
    }
    
    destroyMatrix(m2);
    return 1;
}

//获得数组的元素
double getMatrixElem(Matrix *m,int dimen0,int dimen1,int dimen2,int dimen3){
    return *(m->array + getDiffPointerMaterx(m, dimen0, dimen1, dimen2, dimen3));
}

int modifyMatrixElem(Matrix *m,int dimen0,int dimen1,int dimen2,int dimen3,double elem){
    
    *(m->array + getDiffPointerMaterx(m, dimen0, dimen1, dimen2, dimen3)) = elem;
    return 0;
}
//第二个数组赋值到第一个数组的最后二维
int setMatrixArray(Matrix *m1,Matrix *m2,int dimen0,int dimen1){
    int w,x,y,z,index;
    if(!m1 || !m2) return -1;
    if(m2->dshape.shape[0] !=0 || m2->dshape.shape[1] != 0) return -1;
    z = m1->dshape.shape[3];
    y = m1->dshape.shape[2] * z;
    x = m1->dshape.shape[1] * y;
    w = m1->dshape.shape[0] * x;
    index = dimen0*x + dimen1*y;
    for(int i=0;i<m2->length;i++)
    {
        *(m1->array + index + i) = *(m2->array + i);
    }
    
    return 0;
}
//复制数组
Matrix *copyMatrix(Matrix *m){
    int i = 0;
    if(!m) return NULL;
    Matrix *copym = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!copym) return NULL;
    copym->dshape = m->dshape;
    copym->length = m->length;
    copym->size = m->size;
    copym->array = (double *)MemoryPool_Alloc(copym->length*sizeof(double));
    if(!copym->array){
        MemoryPool_Free(copym);
        return NULL;
    }
    for(i=0;i<copym->length;i++){
        *(copym->array+i) = *(m->array+i);
    }
    return copym;
}
//比较两个数组的维度是否一致
int compareMatrix_Shape(Matrix* m1,Matrix *m2)
{
    int compare_flag = 0;
    if(!m1 || !m2) return 0;
    if(m1->length != m2->length) return 0;
    for(int i=0;i<4;i++)
    {
        if(m1->dshape.shape[i] != m2->dshape.shape[i]){
            compare_flag = 1;
            break;
        }
    }
    if(compare_flag){
        return 0;
    }else{
        return 1;
    }
}
//比较两个数组
int compareMatrix(Matrix *m1, Matrix *m2){
    int i,compare_flag = 0;
    if(!m1 || !m2) return 0;
    if(m1->length != m2->length) return 0;
    for(i=0;i<4;i++)
    {
        if(m1->dshape.shape[i] != m2->dshape.shape[i]){
            compare_flag = 1;
            break;
        }
    }
    if(compare_flag) return 0;
    for(i=0;i<m1->length;i++){
        if(*(m1->array+i) != *(m2->array+i)){
            compare_flag = 1;
            break;
        }
    }
    if(compare_flag){
        return 0;
    }else{
        return 1;
    }
}

//提取二维数组的连续多行，返回一个包含多行内容的子数组
Matrix *getSecondOrderMatrixRows(Matrix *m,int startRow,int endRow){
    int i = 0;
    if(!m) return NULL;
    Matrix *subm = NULL;
    if(m->dshape.shape[0] == 0 && m->dshape.shape[1] == 0 &&
       m->dshape.shape[2] == 0 && m->dshape.shape[3] != 0){
        if(startRow != 0 || endRow != 1) return NULL;
        subm = copyMatrix(m);
    }else if(m->dshape.shape[0] == 0 && m->dshape.shape[1] == 0 &&
             m->dshape.shape[2] != 0 && m->dshape.shape[3] != 0){
        if(startRow < 0 || startRow >= m->dshape.shape[2] ||
           endRow > m->dshape.shape[2] || endRow < startRow) return NULL;
        subm = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
        if(!subm) return NULL;
        if(endRow - startRow == 1){
            subm->dshape.shape[0] = 0;
            subm->dshape.shape[1] = 0;
            subm->dshape.shape[2] = 0;
            subm->dshape.shape[3] = m->dshape.shape[3];
        }else{
            subm->dshape.shape[0] = 0;
            subm->dshape.shape[1] = 0;
            subm->dshape.shape[2] = endRow - startRow;
            subm->dshape.shape[3] = m->dshape.shape[3];
        }
        subm->length = (endRow - startRow)*m->dshape.shape[3];
        subm->size = subm->length;
        subm->array = (double *)MemoryPool_Alloc(subm->length*sizeof(double));
        if(!subm->array){
            free(subm);
            return NULL;
        }
        for(i=0;i<subm->length;i++){
            *(subm->array+i) = *(m->array+startRow*m->dshape.shape[3]+i);
        }
    }
    return subm;
}

//提取二维数组的连续多列，返回一个包含多列内容的子数组
Matrix *getSecondOrderMatrixColumes(Matrix *m,int startColume,int endColume){
    int i = 0,j,w,x,y,z;
    if(!m) return NULL;
    Matrix *subm = NULL;
    if(startColume >= 0 && startColume < m->dshape.shape[3] &&
       endColume <= m->dshape.shape[3] && endColume > startColume){
        subm = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
        if(!subm) return NULL;
        subm->dshape.shape[0] = m->dshape.shape[0];
        subm->dshape.shape[1] = m->dshape.shape[1];
        subm->dshape.shape[2] = m->dshape.shape[2];
        subm->dshape.shape[3] = endColume - startColume;
        z = subm->dshape.shape[3];
        y = z*subm->dshape.shape[2];
        x = y*subm->dshape.shape[1];
        w = x*subm->dshape.shape[0];
        if(w != 0){
            subm->length = w;
        }else if(x != 0){
            subm->length = x;
        }else if(y != 0){
            subm->length = y;
        }else if(z != 0){
            subm->length = z;
        }
        subm->size = subm->length;
        subm->array = (double *)MemoryPool_Alloc(subm->length*sizeof(double));
        if(!subm->array){
            MemoryPool_Free(subm);
            return NULL;
        }
        for(i=0;i<subm->length/z;i++){
            for(j=0;j<subm->dshape.shape[3];j++){
                *(subm->array+i*z+j) = *(m->array+i*m->dshape.shape[3]+startColume+j);
            }
        }
    }
    return subm;
}

//从二维数组一个行与列下标为起点，获取其子数组
Matrix * getSecondOrderSubMatrix(Matrix *m,int startRow,int startColume,int endRow,int endColume){
    Matrix *m1 = NULL,*m2 = NULL;
    m1 = getSecondOrderMatrixRows(m,startRow,endRow);
    if(!m1) return NULL;
    m2 = getSecondOrderMatrixColumes(m1,startColume,endColume);
    destroyMatrix(m1);
    return m2;
}


void getSecondOrderSubMatrix2(Matrix *m,Matrix* mto,int startRow,int startColume){
    //printf("startrow=%d,startcol=%d",startRow,startColume);
    int starti = startColume+startRow * (m->dshape.shape[3]);
    int z = mto->dshape.shape[3];
    for(int i=0;i<mto->dshape.shape[2];i++){
        for(int j=0;j<mto->dshape.shape[3];j++){
            *(mto->array+i*z+j) = *(m->array + starti + i*m->dshape.shape[3]+j);
            //printf("i=%d,j=%d,a=%d,b=%d\n",i,j,i*z + j,starti);
        }
    }
}
void getSecondOrderSubMatrix4d(Matrix *m,Matrix* mto,int dim0,int dim1,int startRow,int startColume)
{
    //printf("startrow=%d,startcol=%d",startRow,startColume);
    int starti_m = getDiffPointerMaterx(m, dim0, dim1, startRow, startColume);// startColume+startRow * (m->dshape.shape[3]);
    //int starti_mto = getDiffPointerMaterx(mto, dim0, dim1, startRow, startColume);
    int z = mto->dshape.shape[3];
    for(int i=0;i<mto->dshape.shape[2];i++){
        for(int j=0;j<mto->dshape.shape[3];j++){
            *(mto->array+i*z+j) = *(m->array + starti_m + i*m->dshape.shape[3]+j);
            //printf("i=%d,j=%d,a=%d,b=%d\n",i,j,i*z + j,starti);
        }
    }
}

//提取二维数组删除指定行和列后的子数组
Matrix *getSecondOrderLeftSubMatrix(Matrix *m,int row,int colume){
    Matrix *copym = NULL;
    copym = copyMatrix(m);
    if(!copym) return NULL;
    if(deleteSecondOrderMatrixRowAndColume(copym,row,colume) == -1) return NULL;
    return copym;
}

//转置二维数组
int transposeSecondOrderMatrix(Matrix *m){
    int i,j,k=0;
    if(!m) return -1;
    Matrix *m1 = copyMatrix(m);
    if(m1->dshape.shape[2] != 0){ //二维数组
        m->dshape.shape[2] = m1->dshape.shape[3];
        m->dshape.shape[3] = m1->dshape.shape[2];
        for(i=0;i<m1->dshape.shape[3];i++){
            for(j=0;j<m1->dshape.shape[2];j++){
                *(m->array+k) = *(m1->array+j*m1->dshape.shape[3]+i);
                k++;
            }
        }
    }else{ //一维数组
        m->dshape.shape[2] = m1->dshape.shape[3];
        m->dshape.shape[3] = 1;
    }
    destroyMatrix(m1);
    return 0;
}

int rot90Matrix(Matrix *m,int d)
{
    if(!m) return 0;
    int z,y,x,w;
    Matrix *m1 = copyMatrix(m);
    if(d%2 == 1)
        transposeSecondOrderMatrix(m);
    z = m->dshape.shape[3];
    y = z*m->dshape.shape[2];
    x = y*m->dshape.shape[1];
    w = x*m->dshape.shape[0];
    for(int i = 0;i< m->dshape.shape[2];i++)
    {
        for(int j =0;j< m->dshape.shape[3];j++)
        {
            if(d == 1)//90度
                *(m->array + j * z + (z-i-1)) = *(m1->array + i * z + j);
            else if (d == 2)//180度
                *(m->array + (z - i - 1) * z + (z - j - 1)) = *(m1->array + i * z + j);
        }
    }
    destroyMatrix(m1);
    return 1;
}

//判断是否实对称矩阵
int isSymmetricMatrix(Matrix *m){
    Matrix *temp = NULL;
    if(!m) return 0;
    //判断是否为二维方阵
    if(m->dshape.shape[0] != 0 || m->dshape.shape[1] != 0 ||
       m->dshape.shape[2] == 0 || m->dshape.shape[2] != m->dshape.shape[3]) return 0;
    temp = copyMatrix(m);
    if(!temp) return 0;
    if(transposeSecondOrderMatrix(temp) != 0) return 0;
    if(compareMatrix(m, temp) == 0){
        destroyMatrix(temp);
        return 0;
    }else{
        destroyMatrix(temp);
        return 1;
    }
}

//交换二维数组的两行
int swapSecondOrderMatrixRow(Matrix *m, int row1,int row2){
    int i;
    if(!m) return -1;
    if(row1 == row2 || m->dshape.shape[2] == 0) return -1;
    if(row1 < 0 || row1 >= m->dshape.shape[2] ||
       row2 < 0 || row2 >= m->dshape.shape[2]) return -1;
    double temp;
    for(i=0;i<m->dshape.shape[3];i++){
        temp = *(m->array+row1*m->dshape.shape[3]+i);
        *(m->array+row1*m->dshape.shape[3]+i) = *(m->array+row2*m->dshape.shape[3]+i);
        *(m->array+row2*m->dshape.shape[3]+i) = temp;
    }
    return 0;
}

//交换二维数组的两列
int swapSecondOrderMatrixColume(Matrix *m, int colume1,int colume2){
    int i;
    if(!m) return -1;
    if(colume1 == colume2) return -1;
    if(colume1 < 0 || colume1 >= m->dshape.shape[3] ||
       colume2 < 0 || colume2 >= m->dshape.shape[3]) return -1;
    double temp;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        temp = *(m->array+i*m->dshape.shape[3]+colume1);
        *(m->array+i*m->dshape.shape[3]+colume1) = *(m->array+i*m->dshape.shape[3]+colume2);
        *(m->array+i*m->dshape.shape[3]+colume2) = temp;
    }
    return 0;
}

//二维数组的其中一行加上一个系数
int kAddSecondOrderMatrixRow(Matrix *m, int row,double k){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0){//一维数组
        if(row != 0) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+i) *= k;
        }
    }else{
        if(row < 0 || row >= m->dshape.shape[2]) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+row*m->dshape.shape[3]+i) += k;
        }
    }
    return 0;
}

//二维数组的其中一行减去一个系数
int kSubSecondOrderMatrixRow(Matrix *m, int row,double k){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0){//一维数组
        if(row != 0) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+i) *= k;
        }
    }else{
        if(row < 0 || row >= m->dshape.shape[2]) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+row*m->dshape.shape[3]+i) -= k;
        }
    }
    return 0;
}

//二维数组的其中一行乘上一个系数
int kMulSecondOrderMatrixRow(Matrix *m, int row,double k){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0){//一维数组
        if(row != 0) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+i) *= k;
        }
    }else{
        if(row < 0 || row >= m->dshape.shape[2]) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+row*m->dshape.shape[3]+i) *= k;
        }
    }
    return 0;
}

//二维数组的其中一行除以一个系数
int kDivSecondOrderMatrixRow(Matrix *m, int row,double k){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0){//一维数组
        if(row != 0) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+i) *= k;
        }
    }else{
        if(row < 0 || row >= m->dshape.shape[2]) return -1;
        for(i=0;i<m->dshape.shape[3];i++){
            *(m->array+row*m->dshape.shape[3]+i) /= k;
        }
    }
    return 0;
}

//二维数组的其中一行加上另外一行,row1 = row1+row2
int addSecondOrderMatrixRows(Matrix *m, int row1,int row2){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0) return -1;
    if(row1 < 0 || row1 >= m->dshape.shape[2] ||
       row2 < 0 || row2 >= m->dshape.shape[2] || row1 == row2) return -1;
    for(i=0;i<m->dshape.shape[3];i++){
        *(m->array+row1*m->dshape.shape[3]+i) += *(m->array+row2*m->dshape.shape[3]+i);
    }
    return 0;
}

//二维数组的其中一行减去另外一行,row1 = row1-row2
int subSecondOrderMatrixRows(Matrix *m, int row1,int row2){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0) return -1;
    if(row1 < 0 || row1 >= m->dshape.shape[2] ||
       row2 < 0 || row2 >= m->dshape.shape[2] || row1 == row2) return -1;
    for(i=0;i<m->dshape.shape[3];i++){
        *(m->array+row1*m->dshape.shape[3]+i) -= *(m->array+row2*m->dshape.shape[3]+i);
    }
    return 0;
}

//二维数组的其中一行乘上另外一行，对应相乘,row1 = row1*row2
int mulSecondOrderMatrixRows(Matrix *m, int row1,int row2){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0) return -1;
    if(row1 < 0 || row1 >= m->dshape.shape[2] ||
       row2 < 0 || row2 >= m->dshape.shape[2] || row1 == row2) return -1;
    for(i=0;i<m->dshape.shape[3];i++){
        *(m->array+row1*m->dshape.shape[3]+i) *= *(m->array+row2*m->dshape.shape[3]+i);
    }
    return 0;
}

//二维数组的其中一行除以另外一行，对应相除,row1 = row1/row2
int divSecondOrderMatrixRows(Matrix *m, int row1,int row2){
    int i;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0) return -1;
    if(row1 < 0 || row1 >= m->dshape.shape[2] ||
       row2 < 0 || row2 >= m->dshape.shape[2] || row1 == row2) return -1;
    for(i=0;i<m->dshape.shape[3];i++){
        *(m->array+row1*m->dshape.shape[3]+i) /= *(m->array+row2*m->dshape.shape[3]+i);
    }
    return 0;
}

//二维数组的其中一列加上一个系数
int kAddSecondOrderMatrixColume(Matrix *m, int colume,double k){
    int i;
    if(!m) return -1;
    if(colume < 0 || colume >= m->dshape.shape[3]) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume) += k;
    }
    return 0;
}

//二维数组的其中一列减去一个系数
int kSubSecondOrderMatrixColume(Matrix *m, int colume,double k){
    int i;
    if(!m) return -1;
    if(colume < 0 || colume >= m->dshape.shape[3]) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume) -= k;
    }
    return 0;
}

//二维数组的其中一列乘上一个系数
int kMulSecondOrderMatrixColume(Matrix *m, int colume,double k){
    int i;
    if(!m) return -1;
    if(colume < 0 || colume >= m->dshape.shape[3]) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume) *= k;
    }
    return 0;
}

//二维数组的其中一列除以一个系数
int kDivSecondOrderMatrixColume(Matrix *m, int colume,double k){
    int i;
    if(!m) return -1;
    if(colume < 0 || colume >= m->dshape.shape[3]) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume) /= k;
    }
    return 0;
}

//二维数组的其中一列加上另外一列,colume1 = colume1 + colume2
int addSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2){
    int i;
    if(!m) return -1;
    if(colume1 < 0 || colume1 >= m->dshape.shape[3] || colume2 < 0 ||
       colume2 >= m->dshape.shape[3] || colume1 == colume2) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume1) += *(m->array+i*m->dshape.shape[3]+colume2);
    }
    return 0;
}

//二维数组的其中一列减去另外一列,colume1 = colume1 - colume2
int subSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2){
    int i;
    if(!m) return -1;
    if(colume1 < 0 || colume1 >= m->dshape.shape[3] || colume2 < 0 ||
       colume2 >= m->dshape.shape[3] || colume1 == colume2) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume1) -= *(m->array+i*m->dshape.shape[3]+colume2);
    }
    return 0;
}

//二维数组的其中一列乘上另外一列,对应相乘,colume1 = colume1 * colume2
int mulSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2){
    int i;
    if(!m) return -1;
    if(colume1 < 0 || colume1 >= m->dshape.shape[3] || colume2 < 0 ||
       colume2 >= m->dshape.shape[3] || colume1 == colume2) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume1) *= *(m->array+i*m->dshape.shape[3]+colume2);
    }
    return 0;
}

//二维数组的其中一列除以另外一列,对应相乘,colume1 = colume1 / colume2
int divSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2){
    int i;
    if(!m) return -1;
    if(colume1 < 0 || colume1 >= m->dshape.shape[3] || colume2 < 0 ||
       colume2 >= m->dshape.shape[3] || colume1 == colume2) return -1;
    for(i=0;i<m->length/m->dshape.shape[3];i++){
        *(m->array+i*m->dshape.shape[3]+colume1) /= *(m->array+i*m->dshape.shape[3]+colume2);
    }
    return 0;
}

//删除二维数组的连续几行
int deleteSecondOrderMatrixRows(Matrix *m,int startRow,int endRow){
    int i = 0;
    if(!m) return -1;
    if(m->dshape.shape[0] == 0 && m->dshape.shape[1] == 0 &&
       m->dshape.shape[2] == 0 && m->dshape.shape[3] != 0){
        if(startRow != 0 || endRow != 1) return -1;
        clearMatrix(m);
    }else if(m->dshape.shape[0] == 0 && m->dshape.shape[1] == 0 &&
             m->dshape.shape[2] != 0 && m->dshape.shape[3] != 0){
        if(startRow < 0 || startRow >= m->dshape.shape[2] ||
           endRow > m->dshape.shape[2] || endRow < startRow) return -1;
        for(i=0;i<m->length-endRow*m->dshape.shape[3];i++){
            *(m->array+startRow*m->dshape.shape[3]+i) = *(m->array+endRow*m->dshape.shape[3]+i);
        }
        m->dshape.shape[2] -= (endRow - startRow);
        if(m->dshape.shape[2] == 1){ //只剩1行，变为一维数组
            m->dshape.shape[2] = 0;
        }
        m->length -= (endRow - startRow)*m->dshape.shape[3];
    }
    return 0;
}

//删除二维数组的连续几列
int deleteSecondOrderMatrixColumes(Matrix *m,int startColume,int endColume){
    if(transposeSecondOrderMatrix(m) == -1) return -1;
    if(deleteSecondOrderMatrixRows(m,startColume,endColume) == -1) return -1;
    return transposeSecondOrderMatrix(m);
}

//删除二维数组的指定的行和列
int deleteSecondOrderMatrixRowAndColume(Matrix *m,int row,int colume){
    if(deleteSecondOrderMatrixColumes(m,colume,colume + 1) == -1) return -1;
    if(deleteSecondOrderMatrixRows(m,row,row + 1) == -1) return -1;
    return 0;
}

//求解数组加上一个系数
int kAddMatrix(Matrix *m,double k){
    int i;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
        *(m->array+i) += k;
    }
    return 0;
}

//两个数组相加，m1,m2的维数必须相同(支持4维)
Matrix *addSecondOrderMatrixs(Matrix *m1,Matrix *m2){
    int i = 0;
    if(compareMatrix_Shape(m1,m2) == 0){printf("m1.ndim != m2.ndim");return NULL;}
    Matrix *resultm = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!resultm) return NULL;
    resultm->size = m1->length;
    resultm->length = resultm->size;
    resultm->array = (double *)MemoryPool_Alloc(resultm->size*sizeof(double));
    if(!resultm->array){
        MemoryPool_Free(resultm);
        return NULL;
    }
    for(i=0;i<resultm->length;i++){
        *(resultm->array + i) = *(m1->array + i) + *(m2->array + i);
    }
    initDshape(&resultm->dshape, m1->dshape.shape);
    return resultm;
}
//两个数组相加，m1,m2的维数必须相同(支持4维)
//不重新申请内存，直接保存到M1中。
void addSecondOrderMatrixs2(Matrix *m1,Matrix *m2){
    if(compareMatrix_Shape(m1,m2) == 0){printf("here   m1.ndim != m2.ndim");return ;}
    for(int i=0;i<m1->length;i++){
        *(m1->array + i) += *(m2->array + i);
    }
}
//两个数组相加，m1,m2的最后两维
//不重新申请内存，直接保存到M1中。
void addSecondOrderMatrixsby2d(Matrix *m1,Matrix *m2,int dim0,int dim1){
    Dshape dshape=m1->dshape;
    if(dshape.shape[2] != m2->dshape.shape[2]
       || dshape.shape[3] != m2->dshape.shape[3]){printf("here   m1.ndim2 != m2.ndim2");return ;}
    int x,y,z,tmp;
    z = dshape.shape[3];
    y = dshape.shape[2] * z;
    x = dshape.shape[1] * y;
    //w = dshape.shape[0] * x;
    
    int startindex = dim0 * x + dim1 * y;
    for(int i =0;i<m1->dshape.shape[2];i++)
    {
        for(int j =0;j<m1->dshape.shape[3];j++)
        {
            tmp = i * z + j;
            *(m1->array + startindex + tmp) += *(m2->array + tmp);
        }
    }
}

//求解数组减去一个系数
int kSubMatrix(Matrix *m,double k){
    int i;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
        *(m->array+i) -= k;
    }
    return 0;
}

//两个数组相减，m1-m2，m1,m2的维数必须相同(支持4维)
Matrix *subSecondOrderMatrixs(Matrix *m1,Matrix *m2){
    int i = 0;
    if(compareMatrix_Shape(m1,m2) == 0){printf("m1.ndim != m2.ndim");return NULL;}
    Matrix *resultm = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!resultm) return NULL;
    resultm->size = m1->length;
    resultm->length = resultm->size;
    resultm->array = (double *)MemoryPool_Alloc(resultm->size*sizeof(double));
    if(!resultm->array){
        MemoryPool_Free(resultm);
        return NULL;
    }
    for(i=0;i<resultm->length;i++){
        *(resultm->array + i) = *(m1->array + i) - *(m2->array + i);
    }
    initDshape(&resultm->dshape, m1->dshape.shape);
    return resultm;
}
//两个数组相减，m1-m2，m1,m2的维数必须相同(支持4维)
void subSecondOrderMatrixs2(Matrix *m1,Matrix *m2)
{
    int i = 0;
    if(compareMatrix_Shape(m1,m2) == 0){printf("m1.ndim != m2.ndim");return ;};
    for(i=0;i<m1->length;i++){
        *((double*)m1->array + i) -= *((double*)m2->array + i);
    }
}

//两个数组点积,对应元素相乘,m1,m2的维数必须相同(支持4维度)
Matrix *dotSecondOrderMatrixs(Matrix *m1,Matrix *m2)
{
    int i = 0;
    if(compareMatrix_Shape(m1,m2) == 0){printf("m1.ndim != m2.ndim");return NULL;}
    Matrix *resultm = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!resultm) return NULL;
    resultm->size = m1->length;
    resultm->length = resultm->size;
    resultm->array = (double *)MemoryPool_Alloc(resultm->size*sizeof(double));
    if(!resultm->array){
        MemoryPool_Free(resultm);
        return NULL;
    }
    for(i=0;i<resultm->length;i++){
        *(resultm->array + i) = *(m1->array + i) * *(m2->array + i);
    }
    initDshape(&resultm->dshape, m1->dshape.shape);
    return resultm;
}
//两个数组点积,对应元素相乘,m1,m2的维数必须相同(支持4维度),不分配内存，保存到m1
void dotSecondOrderMatrixs2(Matrix *m1,Matrix *m2)
{
        int i = 0;
    if(compareMatrix_Shape(m1,m2) == 0){printf("m1.ndim != m2.ndim");return ;};
        for(i=0;i<m1->length;i++){
            *((double*)m1->array + i) *= *((double*)m2->array + i);
        }
}

//求解数组除以一个系数
int kDivMatrix(Matrix *m,double k){
    int i;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
        *(m->array+i) /= k;
    }
    return 0;
}

//求解数组乘一个系数
int kMulMatrix(Matrix *m,double k){
    int i;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
        *(m->array+i) *= k;
    }
    return 0;
}

//两个二维数组的矩阵积，m1为m X n矩阵，m2为n X p矩阵，乘积返回一个m X p矩阵
Matrix *mulSecondOrderMatrixs(Matrix *m1,Matrix *m2){
    int i,j,k,count=0;
    double temp = 0;
    if(!m1 || !m2){
        printf("ERROR mulSecondOrderMatrixs m1 or m2 is NULL!!!!!\n");
        return NULL;
    }
    if(m2->dshape.shape[2] == 0) m2->dshape.shape[2] = 1; //考虑1维数组情况
    if(m1->dshape.shape[3] != m2->dshape.shape[2]){  //矩阵不匹配
        if(m2->dshape.shape[2] == 1) m2->dshape.shape[2] = 0;
        
        printf("ERROR mulSecondOrderMatrixs 矩阵不匹配\n");
        printShape(m1);
        printShape(m2);
        return NULL;
    }
    Matrix *resultm = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!resultm) return NULL;
    if(m1->dshape.shape[2] == 0) m1->dshape.shape[2] = 1; //考虑1维数组情况
    resultm->size = m1->dshape.shape[2] * m2->dshape.shape[3];
    resultm->length = resultm->size;
    resultm->array = (double *)MemoryPool_Alloc(resultm->size*sizeof(double));
    if(!resultm->array){
        MemoryPool_Free(resultm);
        if(m2->dshape.shape[2] == 1) m2->dshape.shape[2] = 0;
        if(m1->dshape.shape[2] == 1) m1->dshape.shape[2] = 0;
        return NULL;
    }
    for(i=0;i<m1->dshape.shape[2];i++){
        for(j=0;j<m2->dshape.shape[3];j++){
            temp = 0;
            for(k=0;k<m1->dshape.shape[3];k++){
                temp += *(m1->array + i*m1->dshape.shape[3]+ k) * *(m2->array + k*m2->dshape.shape[3] + j);
            }
            *(resultm->array + count) = temp;
            count++;
        }
    }
    resultm->dshape.shape[0] = 0;
    resultm->dshape.shape[1] = 0;
    resultm->dshape.shape[2] = m1->dshape.shape[2];
    /*
    if(m2->dshape.shape[2] == 1) m2->dshape.shape[2] = 0;
    if(m1->dshape.shape[2] == 1){
        m1->dshape.shape[2] = 0;
        resultm->dshape.shape[2] = 0;
    }*/
    resultm->dshape.shape[3] = m2->dshape.shape[3];
    return resultm;
}

//使用定义求n阶数组的行列式，要求m为n阶二维方阵，n>=2
static double DetSquareMatrixs(Matrix *m){
    if(!m) return -1;
    if(m->dshape.shape[0] != 0 || m->dshape.shape[1] != 0 ||
       m->dshape.shape[2] != m->dshape.shape[3] || m->dshape.shape[3] < 2) return -1;
    if(m->dshape.shape[3] == 2){
        //递归到一个二维矩阵时可以直接求解行列式
        return *(m->array + 0*m->dshape.shape[2] + 0) * *(m->array + 1*m->dshape.shape[2] + 1) -
        *(m->array + 0*m->dshape.shape[2] + 1) * *(m->array + 1*m->dshape.shape[2] + 0);
    }else{
        double res = 0;
        int i;
        for(i=0;i<m->dshape.shape[2];i++){//为方便编程，每次都由第一行展开
            int k = 1;
            Matrix *subm = NULL;
            if(i % 2 == 1){
                k = -1;
            }
            subm = getSecondOrderLeftSubMatrix(m,0,i);//获取剩下的n-1阶矩阵
            res += *(m->array + i) * k * DetSquareMatrixs(subm);
            destroyMatrix(subm);
        }
        return res;
    }
}

//求行列式，保护传入数组m
int detSquareMatrixs(Matrix *m,double *result){
    Matrix *copym = NULL;
    copym = copyMatrix(m);
    if(!copym) return -1;
    *result = DetSquareMatrixs(copym);
    destroyMatrix(copym);
    return 0;
}

//求二维方阵中指定元素的代数余子式，要求m为n阶二维方阵，n>=2
int getSquareMatrixElemAlgebraicComplement(Matrix *m,int row,int colume,double *result){
    if(!m) return -1;
    if(m->dshape.shape[0] != 0 || m->dshape.shape[1] != 0 ||
       m->dshape.shape[2] != m->dshape.shape[3] || m->dshape.shape[3] < 2) return -1;
    if(row < 0 || row >= m->dshape.shape[2] ||
       colume < 0 || colume >= m->dshape.shape[3]) return -1;
    Matrix *subm = NULL;
    subm = getSecondOrderLeftSubMatrix(m,row,colume);
    if(detSquareMatrixs(subm,result) == -1){
        destroyMatrix(subm);
        return -1;
    }
    if((row+colume)%2 == 1){
        *result *= -1;
    }
    destroyMatrix(subm);
    return 0;
}

//求二维方阵中指定一行元素的代数余子式，要求m为n阶二维方阵，n>=2
Matrix *getSquareMatrixRawAlgebraicComplement(Matrix *m,int row){
    int i;
    double temp;
    if(!m) return NULL;
    if(m->dshape.shape[0] != 0 || m->dshape.shape[1] != 0 ||
       m->dshape.shape[2] != m->dshape.shape[3] || m->dshape.shape[3] < 2) return NULL;
    if(row < 0 || row >= m->dshape.shape[2]) return NULL;
    Matrix *resultm = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!resultm) return NULL;
    resultm->size = m->dshape.shape[3];
    resultm->length = resultm->size;
    resultm->dshape.shape[0] = 0;
    resultm->dshape.shape[1] = 0;
    resultm->dshape.shape[2] = 0;
    resultm->dshape.shape[3] = m->dshape.shape[3];
    resultm->array = (double *)MemoryPool_Alloc(resultm->size*sizeof(double));
    if(!resultm->array){
        MemoryPool_Free(resultm);
        return NULL;
    }
    for(i=0;i<m->dshape.shape[3];i++){
        getSquareMatrixElemAlgebraicComplement(m,row,i,&temp);
        *(resultm->array + i) = temp;
    }
    return resultm;
}

//求二维方阵的伴随矩阵，要求m为n阶二维方阵，n>=2
Matrix *getSquareMatrixAdjointMatrix(Matrix *m){
    int i;
    if(!m) return NULL;
    if(m->dshape.shape[0] != 0 || m->dshape.shape[1] != 0 ||
       m->dshape.shape[2] != m->dshape.shape[3] || m->dshape.shape[3] < 2) return NULL;
    Matrix *resultm = getSquareMatrixRawAlgebraicComplement(m,0);
    for(i=1;i<m->dshape.shape[2];i++){
        Matrix *tempm = NULL;
        tempm = getSquareMatrixRawAlgebraicComplement(m,i);
        spliceSecondOrderMatrixRow(resultm,tempm);
        destroyMatrix(tempm);
    }
    transposeSecondOrderMatrix(resultm);
    return resultm;
}

//寻找方阵的乘法逆矩阵
Matrix *invSquareMatrix(Matrix *m){
    double det_m = 0;
    Matrix *resultm = NULL;
    if(detSquareMatrixs(m,&det_m) == -1) return NULL;
    if(det_m == 0) return NULL; //可逆矩阵行列式不为0
    resultm = getSquareMatrixAdjointMatrix(m);//求伴随矩阵
    if(!resultm) return NULL;
    if(kDivMatrix(resultm,det_m) == -1) return NULL;//逆矩阵=伴随矩阵/行列式
    return resultm;
}

//求m x n二维矩阵的最简阶梯阵,内部使用
static int GetEchelonMatrix(Matrix *m){
    int i,j;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0) return -1;
    for(i=0;i<m->dshape.shape[2]-1;i++){
        if(i >= m->dshape.shape[3]) break; //m>n的情况
        for(j=i+1;j<m->dshape.shape[2];j++){ //保证第一个元素不是0
            if(*(m->array + i*m->dshape.shape[3] + i) == 0 && *(m->array + j*m->dshape.shape[3] + i) != 0){
                swapSecondOrderMatrixRow(m,i,j); //交换两行
                break;
            }
        }
        for(j=i+1;j<m->dshape.shape[2];j++){ //高斯消元法
            double temp = *(m->array + j*m->dshape.shape[3] + i);
            if(temp != 0){
                kMulSecondOrderMatrixRow(m,j,*(m->array + i*m->dshape.shape[3] + i));
                kMulSecondOrderMatrixRow(m,i,temp);
                subSecondOrderMatrixRows(m,j,i);
                kDivSecondOrderMatrixRow(m,i,temp);
            }
        }
    }
    for(i=m->dshape.shape[2]-1;i>=0;i--){
        for(j=i-1;j>=0;j--){ //高斯消元法
            double tempi = *(m->array + i*m->dshape.shape[3] + i);
            double tempj = *(m->array + j*m->dshape.shape[3] + i);
            if(tempi != 0 && tempj != 0){
                kMulSecondOrderMatrixRow(m,j,tempi);
                kMulSecondOrderMatrixRow(m,i,tempj);
                subSecondOrderMatrixRows(m,j,i);
                kDivSecondOrderMatrixRow(m,i,tempj);
            }
        }
    }
    for(i=0;i<m->dshape.shape[2];i++){ //每行首个非0元素置为1
        if(*(m->array + i*m->dshape.shape[3] + i) != 0){
            kDivSecondOrderMatrixRow(m,i,*(m->array + i*m->dshape.shape[3] + i));
        }
    }
    return 0;
}

//求二维矩阵的最简行阶梯阵
Matrix *getEchelonMatrix(Matrix *m){
    int i,j;
    int count0,count1;
    Matrix *copym = NULL;
    copym = copyMatrix(m);
    if(!copym) return NULL;
    if(GetEchelonMatrix(copym) == -1){
        destroyMatrix(copym);
        return NULL;
    }else{
        for(i=0;i<copym->dshape.shape[2]-1;i++){ //全0行移动到后面
            count0 = 0;
            count1 = 0;
            for(j=0;j<copym->dshape.shape[3];j++){
                if(*(copym->array + i*copym->dshape.shape[3] + j) == 0) count0 ++;
                if(*(copym->array + (i+1)*copym->dshape.shape[3] + j) == 0) count1 ++;
            }
            if(count0 == copym->dshape.shape[3] && count1 != copym->dshape.shape[3]){
                swapSecondOrderMatrixRow(copym,i,i+1); //交换两行
            }
        }
        return copym;
    }
}

//求二维矩阵的秩
int getSecondOrderMatrixRank(Matrix *m ,int *rank){
    int i,j,zerocount = 0,zerorow = 0;
    Matrix *tempm = NULL;
    tempm = getEchelonMatrix(m);
    if(!tempm) return -1;
    for(i=0;i<tempm->dshape.shape[2];i++){
        zerocount = 0;
        for(j=0;j<tempm->dshape.shape[3];j++){
            if(*(tempm->array + i*tempm->dshape.shape[3] + j) == 0){
                zerocount ++;
            }
        }
        if(zerocount == tempm->dshape.shape[3]){
            zerorow ++;
        }
    }
    *rank = tempm->dshape.shape[2] - zerorow;
    destroyMatrix(tempm);
    return 0;
}

//求解齐次线性方程组: AX=0，要求A为m X n矩阵，结果为基础解系矩阵
Matrix *solveHomoLinearEquations(Matrix *A){
    int i,j,zerocount = 0,zerorow = 0,rank;
    if(!A) return NULL;
    Matrix *resultm = NULL;
    resultm = copyMatrix(A);
    if(!resultm) return NULL;
    GetEchelonMatrix(resultm);
    for(i=0;i<resultm->dshape.shape[2];i++){
        zerocount = 0;
        for(j=0;j<resultm->dshape.shape[3];j++){
            if(*(resultm->array + i*resultm->dshape.shape[3] + j) == 0){
                zerocount ++;
            }
        }
        if(zerocount == resultm->dshape.shape[3]){
            zerorow ++;
        }
    }
    rank = resultm->dshape.shape[2] - zerorow;
    if(rank == A->dshape.shape[3]){ //r(A) = n，只有0解
        Dshape dshape;
        dshape.shape[0] = 0;
        dshape.shape[1] = 0;
        dshape.shape[2] = A->dshape.shape[3];
        dshape.shape[3] = 1;
        destroyMatrix(resultm);
        resultm = creatZerosMatrix(dshape);
    }else{ //r(A) < n,有非0解
        int columecount = 0;
        for(i=0;i<resultm->dshape.shape[2] ;i++){
            if(*(resultm->array + i*resultm->dshape.shape[3] + i) == 0){
                modifyMatrixElem(resultm,0,0,i,i,-1);
            }
        }
        for(i=resultm->dshape.shape[2];i<resultm->dshape.shape[3];i++){
            Dshape dshape;
            dshape.shape[0] = 0;
            dshape.shape[1] = 0;
            dshape.shape[2] = 0;
            dshape.shape[3] = resultm->dshape.shape[3];
            Matrix *tempm = creatZerosMatrix(dshape);
            modifyMatrixElem(tempm,0,0,0,i,-1);
            spliceSecondOrderMatrixRow(resultm,tempm);
            destroyMatrix(tempm);
        }
        for(i=0;i<resultm->dshape.shape[2] ;i++){
            if(*(resultm->array + i*resultm->dshape.shape[3] + i - columecount) == 1){
                deleteSecondOrderMatrixColumes(resultm,i - columecount,i - columecount + 1);
                columecount++;
            }
        }
        kMulMatrix(resultm,-1);
    }
    return resultm;
}

//求解非齐次线性方程组: AX=B,要求A为m X n矩阵，B为m X 1矩阵,结果为基础解系 + 一个特解(最后一列)
Matrix *solveNonHomoLinearEquations(Matrix *A, Matrix *B){
    int ranka,ranker;
    int i,j,zerocount = 0,zerorow = 0;
    Matrix *erweitertem = NULL;
    Matrix *resultm = NULL;
    Matrix *tempm = NULL;
    if(!A || !B) return NULL;
    if(A->dshape.shape[2] != B->dshape.shape[2]) return NULL;
    erweitertem = copyMatrix(A);
    if(!erweitertem) return NULL;
    spliceSecondOrderMatrixColume(erweitertem,B); //构造增广矩阵
    getSecondOrderMatrixRank(A,&ranka);
    tempm = copyMatrix(erweitertem);
    if(!tempm){
        destroyMatrix(erweitertem);
        return NULL;
    }
    GetEchelonMatrix(tempm);
    for(i=0;i<tempm->dshape.shape[2];i++){
        zerocount = 0;
        for(j=0;j<tempm->dshape.shape[3];j++){
            if(*(tempm->array + i*tempm->dshape.shape[3] + j) == 0){
                zerocount ++;
            }
        }
        if(zerocount == tempm->dshape.shape[3]){
            zerorow ++;
        }
    }
    ranker = tempm->dshape.shape[2] - zerorow;
    destroyMatrix(erweitertem);
    if(ranka < ranker){ //增广矩阵的秩大于A的秩，无解
        destroyMatrix(tempm);
        return NULL;
    }else if(ranka == ranker){ //增广矩阵的秩等于A的秩
        if(ranka == A->dshape.shape[3]){ //r(A) = n,方程组有唯一解
            resultm = getSecondOrderSubMatrix(tempm,0,ranker,ranker,tempm->dshape.shape[3]);
            destroyMatrix(tempm);
            return resultm;
        }else{ //r(A) < n,方程组有无穷多解
            int columecount = 0;
            for(i=0;i<tempm->dshape.shape[2] ;i++){
                if(*(tempm->array + i*tempm->dshape.shape[3] + i) == 0){
                    modifyMatrixElem(tempm,0,0,i,i,-1);
                }
            }
            for(i=tempm->dshape.shape[2];i<tempm->dshape.shape[3]-1;i++){
                Dshape dshape;
                dshape.shape[0] = 0;
                dshape.shape[1] = 0;
                dshape.shape[2] = 0;
                dshape.shape[3] = tempm->dshape.shape[3];
                Matrix *tempm1 = creatZerosMatrix(dshape);
                modifyMatrixElem(tempm1,0,0,0,i,-1);
                spliceSecondOrderMatrixRow(tempm,tempm1);
                destroyMatrix(tempm1);
            }
            for(i=0;i<tempm->dshape.shape[2];i++){
                if(*(tempm->array + i*tempm->dshape.shape[3] + i - columecount) == 1){
                    deleteSecondOrderMatrixColumes(tempm,i - columecount,i - columecount + 1);
                    columecount++;
                }
            }
            for(i=0;i<tempm->dshape.shape[3]-1;i++){
                kMulSecondOrderMatrixColume(tempm,i,-1);
            }
            return tempm;
        }
    }
    return NULL;
}

//雅克比(Jacobi)方法实现实对称矩阵的特征值和特征向量的求解
//返回矩阵第一行为特征值，特征值下面的列为对应的特征向量
Matrix *getSymmetricMatrixEigen(Matrix *m){
    Matrix *resultm = NULL;
    Matrix *tempm = NULL;
    int nCount = 0,i,j;
    if(isSymmetricMatrix(m) == 0) return NULL;
    tempm = copyMatrix(m);
    if(!tempm) return NULL;
    resultm = creatIdentitySecondOrderMatrix(m->dshape);
    if(!resultm) return NULL;
    while(1){
        double dbMax = *(tempm->array + 1);
        int nRow = 0;
        int nCol = 1;
        for(i=0;i<tempm->dshape.shape[2];i++){ //在矩阵非对角线上找到最大的元素
            for(j=0;j<tempm->dshape.shape[3];j++){
                if(i != j){
                    double d = fabs(*(tempm->array + i*tempm->dshape.shape[3] + j)); 
                    if(d > dbMax){
                        dbMax = d;
                        nRow = i;
                        nCol = j;
                    }
                }
            }
        }
        if(dbMax < 1e-10) break; //精度符合要求，不再迭代
        if(nCount > tempm->dshape.shape[3] * tempm->dshape.shape[3] * 30) break; //迭代次数超过限制
        nCount++;
        double dbApp = *(tempm->array + nRow*tempm->dshape.shape[3] + nRow);
        double dbApq = *(tempm->array + nRow*tempm->dshape.shape[3] + nCol);
        double dbAqq = *(tempm->array + nCol*tempm->dshape.shape[3] + nCol);
        
        //计算旋转角度
        double dbAngle = 0.5*atan2(-2*dbApq,dbAqq-dbApp);
        double dbSinTheta = sin(dbAngle);
        double dbCosTheta = cos(dbAngle);
        double dbSin2Theta = sin(2*dbAngle);
        double dbCos2Theta = cos(2*dbAngle);
        
        *(tempm->array + nRow*tempm->dshape.shape[3] + nRow) = dbApp*dbCosTheta*dbCosTheta + 
        dbAqq*dbSinTheta*dbSinTheta + 2*dbApq*dbCosTheta*dbSinTheta;
        *(tempm->array + nCol*tempm->dshape.shape[3] + nCol) = dbApp*dbSinTheta*dbSinTheta + 
        dbAqq*dbCosTheta*dbCosTheta - 2*dbApq*dbCosTheta*dbSinTheta;
        *(tempm->array + nRow*tempm->dshape.shape[3] + nCol) = 0.5*(dbAqq-dbApp)*dbSin2Theta + dbApq*dbCos2Theta;
        *(tempm->array + nCol*tempm->dshape.shape[3] + nRow) = *(tempm->array + nRow*tempm->dshape.shape[3] + nCol);
        
        for(i=0;i<tempm->dshape.shape[3];i++){
            if((i!=nCol)&&(i!=nRow)){
                int u = i*tempm->dshape.shape[3] + nRow; // p
                int w = i*tempm->dshape.shape[3] + nCol; //q
                dbMax = *(tempm->array + u);
                *(tempm->array + u) = *(tempm->array + w) * dbSinTheta + dbMax * dbCosTheta; 
                *(tempm->array + w) = *(tempm->array + w) * dbCosTheta - dbMax * dbSinTheta;
            }
        }
        for(j=0;j<tempm->dshape.shape[3];j++){
            if((j!=nCol)&&(j!=nRow)){
                int u = nRow*tempm->dshape.shape[3] + j; //p
                int w = nCol*tempm->dshape.shape[3] + j; //q
                dbMax = *(tempm->array + u);
                *(tempm->array + u) = *(tempm->array + w) * dbSinTheta + dbMax * dbCosTheta; 
                *(tempm->array + w) = *(tempm->array + w) * dbCosTheta - dbMax * dbSinTheta;
            }
        }
        
        //计算特征向量
        for(i=0;i<resultm->dshape.shape[3];i++){
            int u = i*resultm->dshape.shape[3] + nRow; // p
            int w = i*resultm->dshape.shape[3] + nCol; //q
            dbMax = *(resultm->array + u);
            *(resultm->array + u) = *(resultm->array + w) * dbSinTheta + dbMax*dbCosTheta; 
            *(resultm->array + w) = *(resultm->array + w) * dbCosTheta - dbMax*dbSinTheta;
        }
    }
    Matrix *eigenVal = (Matrix *)MemoryPool_Alloc(sizeof(Matrix));
    if(!eigenVal) return NULL;
    eigenVal->dshape.shape[0] = 0;
    eigenVal->dshape.shape[1] = 0;
    eigenVal->dshape.shape[2] = 0;
    eigenVal->dshape.shape[3] = tempm->dshape.shape[3];
    eigenVal->length = tempm->dshape.shape[3];
    eigenVal->size = eigenVal->length;
    eigenVal->array = (double *)MemoryPool_Alloc(eigenVal->size*sizeof(double));
    if(!eigenVal->array){
        MemoryPool_Free(eigenVal);
        return NULL;
    }
    for(i=0;i<resultm->dshape.shape[3];i++){
        *(eigenVal->array + i) = *(tempm->array + i*tempm->dshape.shape[3] + i);
    }
    spliceSecondOrderMatrixRow(eigenVal,resultm);
    destroyMatrix(tempm);
    destroyMatrix(resultm);
    return eigenVal;
}


//求矩阵的无穷范数,每一行上的元素绝对值求和，再从中取最大的
double getMatrixInfNorm(Matrix *m){
    int i,j;
    double count1=0,count2=0;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0){
        m->dshape.shape[2] = 1;
    }
    for(i=0;i<m->dshape.shape[2];i++){
        count1=0;
        for(j=0;j<m->dshape.shape[3];j++){
            if(*(m->array + i*m->dshape.shape[3] + j) > 0){
                count1 += *(m->array + i*m->dshape.shape[3] + j);
            }else{
                count1 -= *(m->array + i*m->dshape.shape[3] + j);
            }
        }
        if(count1 > count2){
            count2 = count1;
        }
    }
    if(m->dshape.shape[2] == 1){
        m->dshape.shape[2] = 0;
    }
    return count2;
}

//求矩阵的L0范数,非0元素的个数
double getMatrixL0Norm(Matrix *m){
    int i;
    double count = 0;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
        if(*(m->array + i) != 0){
            count += 1;
        }
    }
    return count;
}
    
//每个元素之和
double getMatrixSum(Matrix *m){
    int i;
    double count=0;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
            count += *(double *)(m->array + i);
    }
    return count;
}
double getMatrixMax(Matrix *m,int dim0)
{
    int z = m->dshape.shape[3];
    int y = m->dshape.shape[2] * z ;
    int x = m->dshape.shape[1] * y;
    
    int startindex = dim0 * x;
    double max = *(m->array + startindex),tmp = 0;
    for(int i = 1;i<x;i++)
    {
        tmp = *(m->array + startindex + i);
        if(tmp > max)
        {
            max = tmp;
        }
    }
    return max;
}
//求往后算ndim维度的和
double getMatrixSumbyDim(Matrix *m,int ndim,int dim0,int dim1){
    if(ndim <= 0 || ndim >=4)
        return getMatrixSum(m);
    int z,y,x,w;
    double count=0;
    if(!m) return -1;
    z = m->dshape.shape[3];
    y = z*m->dshape.shape[2];
    x = y*m->dshape.shape[1];
    w = x*m->dshape.shape[0];
    if(ndim ==3)
    {
        for(int i=0;i<m->dshape.shape[1];i++)
        {
            for(int j=0;j<m->dshape.shape[2];j++)
            {
                for(int k=0;k<m->dshape.shape[3];k++)
                {
                    //printf("index = %d\n",dim0 * x + i * y + j * z + k);
                    count += *(double *)(m->array + dim0 * x + i * y + j * z + k);
                }
            }
        }
    }
    else if(ndim == 2)
    {
        for(int i=0;i<m->dshape.shape[2];i++)
        {
            for(int j=0;j<m->dshape.shape[3];j++)
            {
                count += *(double *)(m->array + dim0 * x + dim1 * y + i * z + j);
            }
        }
    }
    return count;
}

//求矩阵的L1范数，每个元素绝对值之和
double getMatrixL1Norm(Matrix *m){
    int i;
    double count=0;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
        if(*(m->array + i) > 0){
            count += *(m->array + i);
        }else{
            count -= *(m->array + i);
        }
    }
    return count;
}

//求矩阵的L2范数,各元素平方之和再开平方根
double getMatrixL2Norm(Matrix *m){
    int i;
    double count=0;
    if(!m) return -1;
    for(i=0;i<m->length;i++){
        count += pow(*(m->array + i),2);
    }
    count = sqrt(count);
    return count;
}

//求矩阵的L21范数,先求各列的L2范数，再将得到的结果求L1范数
double getMatrixL21Norm(Matrix *m){
    int i,j;
    double count1=0,count2=0;
    if(!m) return -1;
    if(m->dshape.shape[2] == 0){
        m->dshape.shape[2] = 1;
    }
    for(i=0;i<m->dshape.shape[3];i++){
        count1=0;
        for(j=0;j<m->dshape.shape[2];j++){
            count1 += pow(*(m->array + j*m->dshape.shape[3] + i),2);
        }
        count1 = sqrt(count1);
        count2 += count1;
    }
    if(m->dshape.shape[2] == 1){
        m->dshape.shape[2] = 0;
    }
    return count2;
}

//数组中如果大于K，则为本身，如果小于K，则为K
void getMaximumMatrix(Matrix *m,double k)
{
    if(!m)return;
    for(int i=0;i<m->length;i++)
    {
        if(*(m->array + i) < k) *(m->array + i) = k;
    }
    return;
}

//每个值求exp
void setMatrixExp(Matrix *m)
{
    if(!m)return;
    for(int i=0;i<m->length;i++)
    {
        *(m->array + i) = exp(*(m->array + i));
    }
    return;
}
//每个值的负数
void setMatrixSign(Matrix *m)
{
    if(!m)return;
    for(int i=0;i<m->length;i++)
    {
        *(m->array + i) = -(*(m->array + i));
    }
    return;
}
void doWise(Matrix *m,double k,...)
{
    if(!m)return;
    va_list argp;
    for(int i=0;i<m->length;i++)
    {
        va_start(argp, k);
        while(1){
            WiseFunc func = va_arg(argp, WiseFunc);
            if(func == NULL) break;
            *(m->array + i) = func(*(m->array + i),k);
        }
        va_end(argp);

         //= func(*(m->array + i),k);
    }
    
    return;
}
