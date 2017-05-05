#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 1024

//__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
//__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__device__ __host__ bool CheckHead(const char* f, const char* c) {
    if(*f == '\n') if(*c != '\n') return true;
    if(*f != '\n') if(*c == '\n') return true;
    return false;
}

__global__ void ParallelScan(const char *text, int *pos, int text_size, int* aux) {
    const int x = threadIdx.x;
    const int bx = blockIdx.x;
    const int attCharIndex = bx * BLOCK_SIZE * 2 + 2 * x;

    __shared__ int temp[2 * BLOCK_SIZE];
    __shared__ int oriTemp[2 * BLOCK_SIZE];
    __shared__ int flag[2 * BLOCK_SIZE];
    __shared__ int oriFlag[2 * BLOCK_SIZE];

    int expo = 1;

    if(attCharIndex >= text_size) {
        temp[2 * x] = 0;
        temp[2 * x + 1] = 0;
        flag[2 * x] = 1;
        flag[2 * x + 1] = 1;
    }
    else if(attCharIndex == text_size - 1) {
        //if(bx != 0 && x == 0 && text[attCharIndex] != '\n') {
        //printf("inhe.\n");
        //    temp[0] = pos[attCharIndex - 1] + 1;
        //}
        //else
        temp[2 * x] = text[attCharIndex] == '\n' ? 0 : 1;
        temp[2 * x + 1] = 0;

        if(x == 0) flag[2 * x] = 1;
        else if(CheckHead(text + attCharIndex - 1, text + attCharIndex)) flag[2 * x] = 1;
        else flag[2 * x] = 0;

        flag[2 * x + 1] = 1;

    }
    else {
        //if(bx != 0 && x == 0 && text[attCharIndex] != '\n') {
        //printf("inhe.\n");
        //    temp[0] = pos[attCharIndex - 1] + 1;
        //}
        //else
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

    /*for test
      __syncthreads();
      if(x == 0 && bx == 0){
      for(int i=0;i<text_size + 8;i++) {
      printf("%d ", temp[i]);
      }
      printf("\n");
      for(int i=0;i<text_size + 8;i++) {
      printf("%d ", flag[i]);
      }
      printf("\n");
      }
     */

    for(int i=1024;i>0;i>>=1) {
        __syncthreads();

        /*for test
          if(x == 0 && bx == 0){
          for(int j=0;j<text_size + 8;j++) {
          printf("%d ", temp[j]);
          }
          printf("\n");
          }
         */

        if(x < i) {
            int ai = expo*(2*x+1) - 1;
            int bi = expo*(2*x+2) - 1;
            if(flag[bi] == 0) temp[bi] += temp[ai];
            flag[bi] |= flag[ai];
        }
        expo *= 2;
    }

    if(x == 0) temp[2047] = 0;

    for(int i=1;i<2048;i*=2) {
        expo /= 2;
        __syncthreads();

        /*for test
          if(x == 0 && bx == 0){
          for(int j=0;j<text_size + 8;j++) {
          printf("%d ", temp[j]);
          }
          printf("\n");
          }
         */

        if(x < i) {
            int ai = expo*(2*x+1) - 1;
            int bi = expo*(2*x+2) - 1;
            int tmp = temp[ai];
            temp[ai] = temp[bi];
            if(oriFlag[ai + 1] == 1) temp[bi] = 0;
            else if(flag[ai] == 1) temp[bi] = tmp;
            else temp[bi] += tmp;
            flag[ai] = 0;
        }
    }

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

    int expo = 1;

    if(attCharIndex >= text_size) {
        temp[2 * x] = 0;
        temp[2 * x + 1] = 0;
        flag[2 * x] = 1;
        flag[2 * x + 1] = 1;
    }
    else if(attCharIndex == text_size - 1) {
        //if(bx != 0 && x == 0 && text[attCharIndex] != '\n') {
        //printf("inhe.\n");
        //    temp[0] = pos[attCharIndex - 1] + 1;
        //}
        //else
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
        //if(bx != 0 && x == 0 && text[attCharIndex] != '\n') {
        //printf("inhe.\n");
        //    temp[0] = pos[attCharIndex - 1] + 1;
        //}
        //else
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

    /*for test
      __syncthreads();
      if(x == 0 && bx == 0){
      for(int i=0;i<text_size + 8;i++) {
      printf("%d ", temp[i]);
      }
      printf("\n");
      for(int i=0;i<text_size + 8;i++) {
      printf("%d ", flag[i]);
      }
      printf("\n");
      }
     */

    for(int i=1024;i>0;i>>=1) {
        __syncthreads();

        /*for test
          if(x == 0 && bx == 0){
          for(int j=0;j<text_size + 8;j++) {
          printf("%d ", temp[j]);
          }
          printf("\n");
          }
         */

        if(x < i) {
            int ai = expo*(2*x+1) - 1;
            int bi = expo*(2*x+2) - 1;
            if(flag[bi] == 0) temp[bi] += temp[ai];
            flag[bi] |= flag[ai];
        }
        expo *= 2;
    }

    if(x == 0) temp[2047] = 0;

    for(int i=1;i<2048;i*=2) {
        expo /= 2;
        __syncthreads();

        /*for test
          if(x == 0 && bx == 0){
          for(int j=0;j<text_size + 8;j++) {
          printf("%d ", temp[j]);
          }
          printf("\n");
          }
         */

        if(x < i) {
            int ai = expo*(2*x+1) - 1;
            int bi = expo*(2*x+2) - 1;
            int tmp = temp[ai];
            temp[ai] = temp[bi];
            if(oriFlag[ai + 1] == 1) temp[bi] = 0;
            else if(flag[ai] == 1) temp[bi] = tmp;
            else temp[bi] += tmp;
            flag[ai] = 0;
        }
    }

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
    //int *cpuAux = new int[text_size];
    cudaMalloc(&aux, sizeof(int) * text_size);
    cudaMemset(aux, 0, sizeof(int) * text_size);
    ParallelScan<<<text_size/1024+1, 1024>>>(text, pos, text_size, aux);
    cudaDeviceSynchronize();
    //	cudaMemcpy(cpuAux, aux, sizeof(int) * text_size, cudaMemcpyDeviceToHost);
    //for(int i=2040;i<2060;i++) printf("%d ", cpuAux[i]);
    //	printf("\n");
    ParallelScanAux<<<text_size/1024+1, 1024>>>(text, pos, text_size, aux);
    //cudaDeviceSynchronize();
    //cudaMemcpy(cpuAux, aux, sizeof(int) * text_size, cudaMemcpyDeviceToHost);
    //for(int i=2040;i<2060;i++) printf("%d ", cpuAux[i]);
    //	printf("\n");
}
