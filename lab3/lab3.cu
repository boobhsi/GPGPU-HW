#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
        const float *background,
        const float *target,
        const float *mask,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        )
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt*yt+xt;
    if (yt < ht and xt < wt and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = target[curt*3+0];
            output[curb*3+1] = target[curt*3+1];
            output[curb*3+2] = target[curt*3+2];
        }
    }
}

__global__ void PoissonImageCloningIteration(const float* fixed, const float* mask, float* input, float* output, int width, int height, int wb, int hb, int ox, int oy) {
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int oneChaDimIdx = (yt * width + xt);
    const int xb = xt + ox;
    const int yb = yt + oy;
    int count = 4;
    if(xb == wb - 1) count -= 1;
    if(xb == 0) count -= 1;
    if(yb == hb - 1) count -= 1;
    if(yb == 0) count -= 1;
    //if(count != 4) printf("nono!\n");
    if(mask[oneChaDimIdx] < 127.0f) return;
    for(int i=0;i<3;i++) {
        int px = oneChaDimIdx * 3 + i;
        float answer = 0.0f;
        if(xt == 0) {
            //no internal
        }
        else if(mask[oneChaDimIdx - 1] > 127.0f) {
            answer += input[px - 3];
        }
        if(xt == width - 1) {
            //no internal
        }
        else if(mask[oneChaDimIdx + 1] > 127.0f) {
            answer += input[px + 3];
        }
        if(yt == 0) {
            //no internal
        }
        else if(mask[oneChaDimIdx - width] > 127.0f) {
            answer += input[px - width * 3];
        }
        if(yt == height - 1) {
            //no internal
        }
        else if(mask[oneChaDimIdx + width] > 127.0f) {
            answer += input[px + width * 3];
        }
        output[px] = (answer + fixed[px]) / count;
    }

}
/*
   __global void JudgeEdge(const float* mask, float* edge, int wt, int ht) {
   const int xt = blockIdx.x * blockDim.x + threadIdx.x;
   const int yt = blockIdx.y * blockDim.y + threadIdx.y;
   const int oneDimIdx = yt * ht + xt;
   if(mask[oneDimIdx] < 127.0f) {
   if(xt != 0) {
   if(mask[oneDimIdx - 1] > 127.0f) {
   edge[oneDimIdx] = 255.0f;
   return;
   }
   }
   else if(xt != wt - 1) {
   if(mask[oneDimIdx + 1] > 127.0f) {
   edge[oneDimIdx] = 255.0f;
   return;
   }
   }
   else if(yt != 0) {
   if(mask[oneDimIdx - wt] > 127.0f) {
   edge[oneDimIdx] = 255.0f;
   return;
   }
   }
   else if(yt != ht - 1) {
   if(mask[oneDimIdx + wt] > 127.0f) {
   edge[oneDimIdx] = 255.0f;
   return;
   }
   }
   }
   edge[oneDimIdx] = 0.0f;
   return;
   }
 */
__global__ void CalculateFixed(const float* background, const float* target, const float* mask, float* fixed, int wb, int hb, int wt, int ht, int oy, int ox) {
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int oneChaDimIdx = (yt * wt + xt);
    const int xb = xt + ox;
    const int yb = yt + oy;
    if(mask[oneChaDimIdx] < 127.0f) return;
    for(int i=0;i<3;i++) {
        int px = oneChaDimIdx * 3 + i;
        int bpx = (yb * wb + xb) * 3 + i;
        float answer = 0.0f;
        //int count = 0;
        if(xt == 0) {
            if(xb != 0) {
                answer += target[px] - 255.0f;
                answer += background[bpx - 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx - 1] < 127.0f) {
                if(xb != 0) {
                    answer += target[px] - target[px - 3];
                    answer += background[bpx - 3];
                }
                else{
                    //no node
                }
            }
            else {
                answer += target[px] - target[px - 3];
            }
        }
        if(xt == wt - 1) {
            if(xb != wb - 1) {
                answer += target[px] - 255.0f;
                answer += background[bpx + 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx + 1] < 127.0f) {
                if(xb != wb - 1) {
                    answer += target[px] - target[px + 3];
                    answer += background[bpx + 3];
                }
                else {
                    //no node
                }
            }
            else {
                answer += target[px] - target[px + 3];
            }
        }
        if(yt == 0) {
            if(yb != 0) {
                answer += target[px] - 255.0f;
                answer += background[bpx - wb * 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx - wt] < 127.0f) {
                if(yb != 0) {
                    answer += target[px] - target[px - wt * 3];
                    answer += background[bpx - wb * 3];
                }
                else {
                    //no node
                }
            }
            else {
                answer += target[px] - target[px - wt * 3];
            }
        }
        if(yt == ht - 1) {
            if(yb != hb - 1) {
                answer += target[px] - 255.0f;
                answer += background[bpx + wb * 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx + wt] < 127.0f) {
                if(yb != hb - 1) {
                    answer += target[px] - target[px + wt * 3];
                    answer += background[bpx + wb * 3];
                }
                else {
                    //no node
                }
            }
            else {
                answer += target[px] - target[px + wt * 3];
            }
        }
        fixed[px] = answer;
    }
}

__global__ ImageDownScaleSampling(float* input, float* output, int scale, int wt, int ht) {
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int afterW = wt / scale;
    const int afterH = ht / scale;
    const int px = yt * afterW + xt;
    const int samplePx = (yt * wt + xt) * scale
    output[px] = input[samplePx];
}

__global__ ImageUpScaleInterpolating(float* input, float* output, int nwt, int nht) {
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int px = yt * nwt + xt;
    const int aPx1 = yt * nwt * 4 * xt * 2;
    const int aPx2 = aPx1 + nwt * 2;
    output[aPx1] = input[px];
    output[aPx1 + 1] = input[px];
    output[aPx2] = input[px];
    output[aPx2 + 1] = input[px];
}

void PoissonImageCloning(
        const float *background,
        const float *target,
        const float *mask,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        )
{
    float *fixed, *buf1, *buf2, *tempMask;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
    cudaMalloc(&tempMask, wt*ht*sizeof(float));

    dim3 gdim8(CeilDiv(wt/8, 32), CeilDiv(ht/8, 16));
    dim3 gdim4(CeilDiv(wt/4, 32), CeilDiv(ht/4, 16));
    dim3 gdim2(CeilDiv(wt/2, 32), CeilDiv(ht/2, 16));
    dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);

    ImageDownScaleSampling(mask, tempMask, 8, wt, ht);
    ImageDownScaleSampling(target, buf1, 8, wt, ht);

    CalculateFixed<<<gdim, bdim>>>(
            background, target, mask, fixed,
            wb, hb, wt, ht, oy, ox
            );

    cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

    for(int i=0; i<10000; i++) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(
                fixed, mask, buf1, buf2, wt, ht, wb, hb, ox, oy
                );
        PoissonImageCloningIteration<<<gdim, bdim>>>(
                fixed, mask, buf2, buf1, wt, ht, wb, hb, ox, oy
                );
    }



    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    SimpleClone<<<gdim, bdim>>>(
            background, buf1, mask, output,
            wb, hb, wt, ht, oy, ox
            );

    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
}
