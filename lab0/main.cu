#include <cstdio>
#include <cstdlib>
#include "../utils/SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

const int W = 40;
const int H = 12;

__global__ void Draw(char *frame) {
	// TODO: draw more complex things here
	// Do not just submit the original file provided by the TA!
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H and x < W) {
		char c;
		if (x == W-1) {
			c = y == H-1 ? '\0' : '\n';
		} else if (y == 0 or y == H-1 or x == 0 or x == W-2) {
			c = ':';
		} else {
            bool element_drawed = false;
            if (y <= 10 && y >= 5) {
                if (x <= 21 && x >= 8+(10-y)*2) {
                    c = '#';
                    element_drawed = true;
                } else if (x == 33) {
                    if (y == 10) {
                        c = '#';
                        element_drawed = true;
                    } else {
                        c = '|';
                        element_drawed = true;
                    }
                } else if (x == 32 && y == 5) {
                    c = '<';
                    element_drawed = true;
                }
            }
			if (!element_drawed) c = ' ';
		}
		frame[y*W+x] = c;
	}
}

int main(int argc, char **argv)
{
	MemoryBuffer<char> frame(W*H);
	auto frame_smem = frame.CreateSync(W*H);
	CHECK;

	Draw<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(frame_smem.get_gpu_wo());
	CHECK;

	puts(frame_smem.get_cpu_ro());
	CHECK;
	return 0;
}
