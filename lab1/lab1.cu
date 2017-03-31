#include "lab1.h"
#include <cmath>
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_FUNCTION __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_FUNCTION
#endif

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

const unsigned blockSizeX = 20;
const unsigned blockSizeY = 20;

template <class T>
class Vector {
public:
    CUDA_CALLABLE_MEMBER Vector(T x, T y, T z);
    CUDA_CALLABLE_MEMBER Vector();
    CUDA_CALLABLE_MEMBER ~Vector();
    CUDA_CALLABLE_MEMBER void mod(const unsigned u);
    CUDA_CALLABLE_MEMBER void set(T xi, T yi, T zi);

    T coor[3];
};

CUDA_FUNCTION void Cuda_noise(uint8_t*, unsigned);
CUDA_CALLABLE_MEMBER float inter(float, float, float);
template<class T>
CUDA_CALLABLE_MEMBER float gradient(uint8_t, Vector<T>&);
CUDA_CALLABLE_MEMBER float fade(float);
template<class T>
CUDA_CALLABLE_MEMBER void distance(Vector<T>&, Vector<T>&);

template<class T>
CUDA_CALLABLE_MEMBER Vector<T>::Vector(T x, T y, T z) {
    coor[0] = x;
    coor[1] = y;
    coor[2] = z;
}

template<class T>
CUDA_CALLABLE_MEMBER Vector<T>::Vector() {}

template<class T>
CUDA_CALLABLE_MEMBER Vector<T>::~Vector() {}

template<class T>
CUDA_CALLABLE_MEMBER void Vector<T>::mod(const unsigned u) {
    this->set((int)coor[0] % u, (int)coor[1] % u, (int)coor[2] % u);
}

template<class T>
CUDA_CALLABLE_MEMBER void Vector<T>::set(T xi, T yi, T zi) {
    coor[0] = xi;
    coor[1] = yi;
    coor[2] = zi;
}

struct Lab1VideoGenerator::Impl {
    int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
    info.w = W;
    info.h = H;
    info.n_frame = NFRAME;
    // fps = 24/1 = 24
    info.fps_n = 24;
    info.fps_d = 1;
};

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
    Cuda_noise<<<dim3(W/blockSizeX, H/blockSizeY), dim3(blockSizeX, blockSizeY)>>>(yuv, impl->t);
    ++(impl->t);

}

CUDA_FUNCTION void Cuda_noise(uint8_t *yuv, unsigned frameIdx) {

    __shared__ uint8_t permutation[512];
    /*
       cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
       cudaMemset(yuv+W*H, 128, W*H/2);
       ++(impl->t);
     */
    //unsigned frameIdx;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("cuda_called: (%d,%d)\n", x, y);

    if(threadIdx.x==0 && threadIdx.y==0) {
        uint8_t temp[512] =
        {151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
            88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
            77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
            102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
            223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
            129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
            251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
            49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
            151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
            88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
            77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
            102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
            223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
            129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
            251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
            49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        };
        for(int i=0;i<512;i++) {
            permutation[i] = temp[i];
	        // printf("%d: %d\n", i, permutation[i]);
	    }
    }

    bool uvchannel = false;

    if(x%2 == 0 && y%2 == 0) uvchannel = true;

    //Vector<int> cell(blockIdx.x , blockIdx.y , 0);
    Vector<float> pp((float)(x) / 32, (float)(y) / 32, frameIdx * 0.75 / 32);

    Vector<int> cell((int)floor(pp.coor[0]), (int)floor(pp.coor[1]), (int)floor(pp.coor[2]));

    float ux = pp.coor[0] - cell.coor[0];
    float uy = pp.coor[1] - cell.coor[1];

    cell.mod(255);

    float uz[3] = {
        (pp.coor[2] - cell.coor[2]),
        0.0,
        0.0
    };

    for(int i=1;i<1;i++) {
        uz[i] -= floor(uz[i]);
    }

    float answer[3];

    float uxf = fade(ux);
    float uyf = fade(uy);

    Vector<float> point, unit;

    __syncthreads();

    for(size_t i=0; i<1; i++) {
        float uzf = fade(uz[i]);
        unit.set(ux, uy, uz[i]);
        point.set(0.0, 0.0, 0.0);
        distance(unit, point);
        float aaa = gradient(permutation[permutation[permutation[cell.coor[0]]+cell.coor[1]]+cell.coor[2]], point);
        point.set(0.0, 0.0, 1.0);
        distance(unit, point);
        float aab = gradient(permutation[permutation[permutation[cell.coor[0]]+cell.coor[1]]+cell.coor[2]+1], point);
        point.set(0.0, 1.0, 0.0);
        distance(unit, point);
        float aba = gradient(permutation[permutation[permutation[cell.coor[0]]+cell.coor[1]+1]+cell.coor[2]], point);
        point.set(1.0, 0.0, 0.0);
        distance(unit, point);
        float baa = gradient(permutation[permutation[permutation[cell.coor[0]+1]+cell.coor[1]]+cell.coor[2]], point);
        point.set(0.0, 1.0, 1.0);
        distance(unit, point);
        float abb = gradient(permutation[permutation[permutation[cell.coor[0]]+cell.coor[1]+1]+cell.coor[2]+1], point);
        point.set(1.0, 0.0, 1.0);
        distance(unit, point);
        float bab = gradient(permutation[permutation[permutation[cell.coor[0]+1]+cell.coor[1]]+cell.coor[2]+1], point);
        point.set(1.0, 1.0, 0.0);
        distance(unit, point);
        float bba = gradient(permutation[permutation[permutation[cell.coor[0]+1]+cell.coor[1]+1]+cell.coor[2]], point);
        point.set(1.0, 1.0, 1.0);
        distance(unit, point);
        float bbb = gradient(permutation[permutation[permutation[cell.coor[0]+1]+cell.coor[1]+1]+cell.coor[2]+1], point);

        float _ab = inter(aab, bab, uxf);
        float _bb = inter(abb, bbb, uxf);
        float __b = inter(_ab, _bb, uyf);
        float _aa = inter(aaa, baa, uxf);
        float _ba = inter(aba, bba, uxf);
        float __a = inter(_aa, _ba, uyf);
        answer[i] = inter(__a, __b, uzf);
    }

    answer[0] = (answer[0] + 1) / 2 * 255.0;
    answer[1] = (sin(frameIdx / 32.0f) + 1) / 2 * 255.0;
    answer[2] = (cos(frameIdx / 32.0f) + 1) / 2 * 255.0;

    *(yuv+y*W+x) = (uint8_t)answer[0];

    if(uvchannel) {
        int pxuv = x/2;
        int pyuv = y/2;
        *(yuv+W*H+pyuv*W/2+pxuv) = (uint8_t)answer[1];
        *(yuv+W*H+H*W/4+pyuv*W/2+pxuv) = (uint8_t)answer[2];

    }

}

CUDA_CALLABLE_MEMBER float inter(float lo, float  hi, float ref) {
    return (hi - lo) * ref + lo;
}

template<class T>
CUDA_CALLABLE_MEMBER float gradient(uint8_t hash, Vector<T>& dist) {
    int h = hash & 15;
    float u = h < 8 ? dist.coor[0] : dist.coor[1];

    float v;

    if(h < 4) v = dist.coor[1];
    else if(h == 12 || h == 14) v = dist.coor[0];
    else v = dist.coor[2];

    return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}

CUDA_CALLABLE_MEMBER float fade(float x) {
    return  6 * pow(x, 5) - 15 * pow(x, 4) + 10 * pow(x, 3);
}

template<class T>
CUDA_CALLABLE_MEMBER void distance(Vector<T>& des, Vector<T>& sta) {
    sta.set(des.coor[0]-sta.coor[0], des.coor[1]-sta.coor[1], des.coor[2]-sta.coor[2]);
}
