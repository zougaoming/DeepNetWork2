//
//  Link.c
//  GateC
//
//  Created by Steven on 20/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "Link.h"
#include "memorypool.h"
pNode head = NULL;

//display the list
void printList()
{
    pNode ptr = head;
    printf("\n[ ");
    
    //start from the beginning
    while (ptr != NULL)
    {
        printf("(%d) ", ptr->key);
        ptr = ptr->next;
    }
    
    printf(" ]\n");
}

void* _Link_insert(pNode paramlist,int key,void *value,int datalen)
{
    pNode newp = (pNode)MemoryPool_Alloc(sizeof(Node));
    newp->key = key;
    newp->next = NULL;
    newp->data = MemoryPool_Alloc(datalen);
    memcpy(newp->data,value,datalen);
    paramlist->next = newp;
    return newp;
}

pNode findAndCreateLink(int needSave,int key,void* value,int datalen)
{
    if(head == NULL){
        head = (pNode)MemoryPool_Alloc(sizeof(Node));
        head->key = key;
        head->data = MemoryPool_Alloc(datalen);
        memcpy(head->data,value,datalen);
        head->next = NULL;
        return head;
    }
    pNode curp = head;
    pNode lastp = head;
    while(curp != NULL)
    {
        if(curp->key == key)
        {
            if(needSave == 1){
                memcpy(curp->data,value,datalen);
            }
            return curp;
        }
        lastp = curp;
        curp = curp->next;
    }
    return _Link_insert(lastp,key,value,datalen);
}
