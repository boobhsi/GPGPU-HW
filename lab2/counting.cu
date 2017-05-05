#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 1024

__device__ bool CheckHead(const char* f, const char* c) {
    if(*f == '\n') if(*c != '\n') return true;
    if(*f != '\n') if(*c == '\n') return true;
    return false;
}

__device__ void SegmentedScan(int* target, int* flag, int* oriFlag, const int tx) {
    int expo = 1;
    for(int i=BLOCK_SIZE;i>0;i>>=1) {
        __syncthreads();
        if(tx < i) {
            int ai = expo*(2*tx+1) - 1;
            int bi = expo*(2*tx+2) - 1;
            if(flag[bi] == 0) target[bi] += target[ai];
            flag[bi] |= flag[ai];
        }
        expo *= 2;
    }

    if(tx == 0) target[BLOCK_SIZE * 2 - 1] = 0;

    for(int i=1;i<BLOCK_SIZE*2;i*=2) {
        expo /= 2;
        __syncthreads();
        if(tx < i) {
            int ai = expo*(2*tx+1) - 1;
            int bi = expo*(2*tx+2) - 1;
            int tmp = target[ai];
            target[ai] = target[bi];
            if(oriFlag[ai + 1] == 1) target[bi] = 0;
            else if(flag[ai] == 1) target[bi] = tmp;
            else target[bi] += tmp;
            flag[ai] = 0;
        }
    }
}

__global__ void ParallelScanWord(const char *text, int *pos, int text_size, int* aux) {
    const int x = threadIdx.x;
    const int bx = blockIdx.x;
    const int attCharIndex = bx * BLOCK_SIZE * 2 + 2 * x;

    __shared__ int temp[2 * BLOCK_SIZE];
    __shared__ int oriTemp[2 * BLOCK_SIZE];
    __shared__ int flag[2 * BLOCK_SIZE];
    __shared__ int oriFlag[2 * BLOCK_SIZE];

    if(attCharIndex >= text_size) {
        temp[2 * x] = 0;
        temp[2 * x + 1] = 0;
        flag[2 * x] = 1;
        flag[2 * x + 1] = 1;
    }
    else if(attCharIndex == text_size - 1) {
        temp[2 * x] = text[attCharIndex] == '\n' ? 0 : 1;
        temp[2 * x + 1] = 0;

        if(x == 0) flag[2 * x] = 1;
        else if(CheckHead(text + attCharIndex - 1, text + attCharIndex)) flag[2 * x] = 1;
        else flag[2 * x] = 0;

        flag[2 * x + 1] = 1;
    }
    else {
        temp[2 * x] = text[attCharIndex] == '\n' ? 0 : 1;
        temp[2 * x + 1] = text[attCharIndex + 1] == '\n' ? 0 : 1;

        if(x == 0) flag[2 * x] = 1;
        else if(CheckHead(text + attCharIndex - 1, text + attCharIndex)) flag[2 * x] = 1;
        else flag[2 * x] = 0;

        if(CheckHead(text + attCharIndex + 1, text + attCharIndex)) flag[2 * x + 1] = 1;
        else flag[2 * x + 1] = 0;
    }

    oriFlag[2 * x] = flag[2 * x];
    oriFlag[2 * x + 1] = flag[2 * x + 1];

    oriTemp[2 * x] = temp[2 * x];
    oriTemp[2 * x + 1] = temp[2 * x + 1];

    SegmentedScan(temp, flag, oriFlag, x);

    if(attCharIndex < text_size) {
        pos[attCharIndex] = temp[2 * x] + oriTemp[2 * x];
        if(attCharIndex != text_size - 1) pos[attCharIndex + 1] = temp[2 * x + 1] + oriTemp[2 * x + 1];
    }

    if(x == BLOCK_SIZE - 1 && attCharIndex + 2 < text_size){
        aux[attCharIndex + 2] = temp[2 * x + 1] + oriTemp[2 * x + 1];
    }
}

__global__ void ParallelScanAux(const char *text, int *pos, int text_size, int* aux) {
    const int x = threadIdx.x;
    const int bx = blockIdx.x;
    const int attCharIndex = bx * BLOCK_SIZE * 2 + 2 * x;

    __shared__ int temp[2 * BLOCK_SIZE];
    __shared__ int oriTemp[2 * BLOCK_SIZE];
    __shared__ int flag[2 * BLOCK_SIZE];
    __shared__ int oriFlag[2 * BLOCK_SIZE];


    if(attCharIndex >= text_size) {
        temp[2 * x] = 0;
        temp[2 * x + 1] = 0;
        flag[2 * x] = 1;
        flag[2 * x + 1] = 1;
    }
    else if(attCharIndex == text_size - 1) {
        temp[2 * x] = aux[attCharIndex];
        temp[2 * x + 1] = 0;

        if(bx != 0 && x == 0 && text[attCharIndex - 1] != '\n' && text[attCharIndex] != '\n') flag[2 * x] = 1;
        else {
            flag[2 * x] = 0;
            temp[2 * x] = 0;
        }

        flag[2 * x + 1] = 0;

    }
    else {
        temp[2 * x] = aux[attCharIndex];
        temp[2 * x + 1] = aux[attCharIndex + 1];

        if(bx != 0 && x == 0 ){
            if(text[attCharIndex -1] != '\n' && text[attCharIndex] != '\n') flag[2 * x] = 1;
            else {flag[2 * x] = 0;
                temp[2 * x] = 0;
            }

        }
        else if(CheckHead(text + attCharIndex - 1, text + attCharIndex)) flag[2 * x] = 1;
        else flag[2 * x] = 0;

        if(CheckHead(text + attCharIndex + 1, text + attCharIndex)) flag[2 * x + 1] = 1;
        else flag[2 * x + 1] = 0;

    }

    oriFlag[2 * x] = flag[2 * x];
    oriFlag[2 * x + 1] = flag[2 * x + 1];

    oriTemp[2 * x] = temp[2 * x];
    oriTemp[2 * x + 1] = temp[2 * x + 1];

    SegmentedScan(temp, flag, oriFlag, x);

    if(attCharIndex < text_size) {
        pos[attCharIndex] += (temp[2 * x] + oriTemp[2 * x]);
        if(attCharIndex != text_size - 1) pos[attCharIndex + 1] += (temp[2 * x + 1] + oriTemp[2 * x + 1]);
    }
}

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::device_ptr<const char> dText(text);
    thrust::device_ptr<int> dPos(pos);
    thrust::fill(thrust::device, dPos, dPos + text_size, '\n');
    thrust::transform(thrust::device, dText, dText + text_size, dPos, dPos, thrust::not_equal_to<char>());
    thrust::inclusive_scan_by_key(thrust::device, dPos, dPos + text_size, dPos, dPos, thrust::equal_to<int>());
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int *aux;
    cudaMalloc(&aux, sizeof(int) * text_size);
    cudaMemset(aux, 0, sizeof(int) * text_size);
    ParallelScanWord<<<text_size/1024+1, 1024>>>(text, pos, text_size, aux);
    cudaDeviceSynchronize();
    ParallelScanAux<<<text_size/1024+1, 1024>>>(text, pos, text_size, aux);
}
