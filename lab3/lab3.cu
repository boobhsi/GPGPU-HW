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
    if(xt >= width || yt >= height) return;
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
        else if(mask[oneChaDimIdx - 1] > 127.0f && xb > 0) {
            answer += input[px - 3];
        }
        if(xt == width - 1) {
            //no internal
        }
        else if(mask[oneChaDimIdx + 1] > 127.0f && xb < wb - 1) {
            answer += input[px + 3];
        }
        if(yt == 0) {
            //no internal
        }
        else if(mask[oneChaDimIdx - width] > 127.0f && yb > 0) {
            answer += input[px - width * 3];
        }
        if(yt == height - 1) {
            //no internal
        }
        else if(mask[oneChaDimIdx + width] > 127.0f && yb < hb - 1) {
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
    if(xt >= wt || yt >= ht) return;
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
            if(xb > 0) {
                answer += target[px] - 255.0f;
                answer += background[bpx - 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx - 1] < 127.0f) {
                if(xb > 0) {
                    answer += target[px] - target[px - 3];
                    answer += background[bpx - 3];
                }
                else{
                    //no node
                }
            }
            else {
                if(xb > 0) answer += target[px] - target[px - 3];
            }
        }
        if(xt == wt - 1) {
            if(xb < wb - 1) {
                answer += target[px] - 255.0f;
                answer += background[bpx + 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx + 1] < 127.0f) {
                if(xb < wb - 1) {
                    answer += target[px] - target[px + 3];
                    answer += background[bpx + 3];
                }
                else {
                    //no node
                }
            }
            else {
                if(xb < wb - 1) answer += target[px] - target[px + 3];
            }
        }
        if(yt == 0) {
            if(yb > 0) {
                answer += target[px] - 255.0f;
                answer += background[bpx - wb * 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx - wt] < 127.0f) {
                if(yb > 0) {
                    answer += target[px] - target[px - wt * 3];
                    answer += background[bpx - wb * 3];
                }
                else {
                    //no node
                }
            }
            else {
                if(yb > 0) answer += target[px] - target[px - wt * 3];
            }
        }
        if(yt == ht - 1) {
            if(yb < hb - 1) {
                answer += target[px] - 255.0f;
                answer += background[bpx + wb * 3];
            }
            else {
                //no node
            }
        }
        else {
            if(mask[oneChaDimIdx + wt] < 127.0f) {
                if(yb < hb - 1) {
                    answer += target[px] - target[px + wt * 3];
                    answer += background[bpx + wb * 3];
                }
                else {
                    //no node
                }
            }
            else {
                if(yb < hb - 1) answer += target[px] - target[px + wt * 3];
            }
        }
        fixed[px] = answer;
    }
}

__global__ void ImageDownScaleSampling(const float* input, float* output, int scale, int wt, int ht, int channel) {
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int afterW = wt / scale;
    const int afterH = ht / scale;
    if(xt >= afterW || yt >= afterH) return;
    const int px = (yt * afterW + xt) * channel;
    const int samplePx = ((yt * wt + xt) * scale) * channel;
    for(int i=0;i<channel;i++) output[px+i] = input[samplePx+i];
}

__global__ void ImageUpScaleInterpolating(const float* input, float* output, int nwt, int nht, int channel) {
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    if(xt >= nwt || yt >= nht) return;
    const int px = (yt * nwt + xt) * channel;
    const int aPx1 = (yt * nwt * 4 + xt * 2) * channel;
    const int aPx2 = aPx1 + nwt * 2 * channel;
    for(int i=0;i<channel;i++) {
        output[aPx1+i] = input[px+i];
        output[aPx1 + channel + i] = input[px+i];
        output[aPx2 + i] = input[px+i];
        output[aPx2 + channel + i] = input[px+i];
    }
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

    const int base = 500;
    const bool linear = false;
    const float pow = 5.0f;
    const float bScale = 10.0f;
    const int start = 0;
    const int testLimit = 4;


    float *fixed, *buf1, *buf2, *tempMask, *upTempMask;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
    cudaMalloc(&tempMask, wt*ht*sizeof(float));
    cudaMalloc(&upTempMask, wt*ht*sizeof(float));

    dim3 *gdim = new dim3[4];
    gdim[0] = dim3(CeilDiv(wt/8, 32), CeilDiv(ht/8, 16));
    //printf("%d, %d\n", wt/8, ht/8);
    gdim[1] = dim3(CeilDiv(wt/4, 32), CeilDiv(ht/4, 16));
    gdim[2] = dim3(CeilDiv(wt/2, 32), CeilDiv(ht/2, 16));
    gdim[3] = dim3(CeilDiv(wt, 32), CeilDiv(ht, 16));
    dim3 bdim(32, 16);

    //if(linear) printf("linear mode\n");
    //else printf("power mode\n");

    //int testLimit = 4;
    //int start = 0; 
    int scale;

    for(int i=start;i<testLimit;i++) {

        scale = 8 / powf(2, i);;
	
	//cudaMemset(tempMask, 0, wt*ht*sizeof(float));
	//cudaMemset(buf1, 255, wt*ht*sizeof(float)*3);

	if(testLimit == 4 && start == 3) ;//do nothing
	else if(i == start) ImageDownScaleSampling<<<gdim[i], bdim>>>(mask, tempMask, scale, wt, ht, 1);
	else cudaMemcpy(tempMask, upTempMask, wt*ht*sizeof(float), cudaMemcpyDeviceToDevice);
	
	if(testLimit == 4 && start == 3) cudaMemcpy(buf1, target, wt*ht*3*sizeof(float), cudaMemcpyDeviceToDevice);
        else if(i != 3) ImageDownScaleSampling<<<gdim[i], bdim>>>(target, buf1, scale, wt, ht, 3);

	
	if(start == 3 && testLimit == 4) {
	    CalculateFixed<<<gdim[3], bdim>>>(
		background, target, mask, fixed,
                wb, hb, wt, ht, oy, ox
                );
	}
	else if(i != 3) {
    	   CalculateFixed<<<gdim[i], bdim>>>(
		background, buf1, tempMask, fixed,
                wb, hb, wt/scale, ht/scale, oy, ox
                );
        }
	else {
	   CalculateFixed<<<gdim[i], bdim>>>(
                background, target, tempMask, fixed,
                wb, hb, wt, ht, oy, ox
                );
 	}

	if(i != start) {
	    cudaMemcpy(buf1, buf2, wt*ht*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	}

	for(int k=0; k < (linear ? base * (i - start + 1) : base * (logf(i*bScale+1)/logf(pow)+1)/* powf(i, )*/); k++) {
            PoissonImageCloningIteration<<<gdim[i], bdim>>>(
                fixed, ((start == 3 && testLimit == 4) ? mask : tempMask), buf1, buf2, wt/scale, ht/scale, wb, hb, ox, oy
                );
            PoissonImageCloningIteration<<<gdim[i], bdim>>>(
                fixed, ((start == 3 && testLimit == 4) ? mask : tempMask), buf2, buf1, wt/scale, ht/scale, wb, hb, ox, oy
                );
        }

        if(i != testLimit - 1) {
	    ImageUpScaleInterpolating<<<gdim[i], bdim>>>(buf1, buf2, wt/scale, ht/scale, 3);
	    ImageUpScaleInterpolating<<<gdim[i], bdim>>>(tempMask, upTempMask, wt/scale, ht/scale, 1);
	}

    }

    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

    if(start == 3 && testLimit == 4) 
	SimpleClone<<<gdim[3], bdim>>>(
            background, buf1, mask, output,
            wb, hb, wt, ht, oy, ox
            );
    else 
	SimpleClone<<<gdim[testLimit - 1], bdim>>>(
            background, buf1, tempMask, output,
            wb, hb, wt/scale, ht/scale, oy, ox
            );


    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
    cudaFree(tempMask);
    cudaFree(upTempMask);
}
