//
//  Link.h
//  GateC
//
//  Created by Steven on 20/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef Link_h
#define Link_h

#include <stdio.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Node_TAG
{
    int key;
    void* data;
    unsigned int dataLen;
    struct Node_TAG * next;
}Node,*pNode;
typedef enum KeyType_UpdateParam_TAG{
    w1 = 100,
    w2,
    w3,
    b1,
    b2,
    b3,
    adam_t,
}updateParamKey;
pNode findAndCreateLink(int needSave,int key,void* value,int datalen);

#endif /* Link_h */
