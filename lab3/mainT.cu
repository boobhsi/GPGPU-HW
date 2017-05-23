#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include "../utils/SyncedMemory.h"
#include "../utils/Timer.h"
#include "pgm.h"
#include "lab3.h"
#include "fstream"
using namespace std;

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

int main(int argc, char **argv)
{
	if (argc != 13) {
		printf("Usage: %s <background> <target> <mask> <offset x> <offset y> <output> <linear(0 for false, 1 for true)> <base> <power_base> <log_x_base> <start_scale(0 for 1/8, 3 for original)> <scale_terminal(1 for 1/8, 4 for original)>\n", argv[0]);
		abort();
	}
	bool sucb, suct, sucm;
	int wb, hb, cb, wt, ht, ct, wm, hm, cm;
    int wtest, htest, ctest;
    bool suctest;
	auto imgb = ReadNetpbm(wb, hb, cb, sucb, argv[1]);
	auto imgt = ReadNetpbm(wt, ht, ct, suct, argv[2]);
	auto imgm = ReadNetpbm(wm, hm, cm, sucm, argv[3]);
    auto imgTest = ReadNetpbm(wtest, htest, ctest, suctest, "lab3_test/output.ppm");
	if (not (sucb and suct and sucm)) {
		puts("Something wrong with reading the input image files.");
		abort();
	}
	if (wt != wm or ht != hm) {
		puts("The mask and target image must have the same size.");
		abort();
	}
	if (cm != 1) {
		puts("The mask image must be mono-colored.");
		abort();
	}
	if (cb != 3 or ct != 3) {
		puts("The background and target image must be colored.");
		abort();
	}

    fstream file;
    file.open("test.csv", ios::out|ios::app);
    if(!file) {
        cerr<<"test file generator error!\n";
        exit(1);
    }

	const int oy = atoi(argv[4]), ox = atoi(argv[5]);
	const int linear = atoi(argv[7]), base = atoi(argv[8]), start = atoi(argv[11]), testL = atoi(argv[12]);
	const float dre = atof(argv[9]), bScale = atof(argv[10]);

	const int SIZEB = wb*hb*3;
	const int SIZET = wt*ht*3;
	const int SIZEM = wm*hm;
	MemoryBuffer<float> background(SIZEB), target(SIZET), mask(SIZEM), output(SIZEB);
	auto background_s = background.CreateSync(SIZEB);
	auto target_s = target.CreateSync(SIZET);
	auto mask_s = mask.CreateSync(SIZEM);
	auto output_s = output.CreateSync(SIZEB);

	float *background_cpu = background_s.get_cpu_wo();
	float *target_cpu = target_s.get_cpu_wo();
	float *mask_cpu = mask_s.get_cpu_wo();
	copy(imgb.get(), imgb.get()+SIZEB, background_cpu);
	copy(imgt.get(), imgt.get()+SIZET, target_cpu);
	copy(imgm.get(), imgm.get()+SIZEM, mask_cpu);

    Timer timer;

	if(linear == 0) {
        timer.Start();
	    PoissonImageCloning(
		background_s.get_gpu_ro(),
		target_s.get_gpu_ro(),
		mask_s.get_gpu_ro(),
		output_s.get_gpu_wo(),
		wb, hb, wt, ht, oy, ox,
		base, false, dre, bScale, start, testL
	    );
    }
	else {
        timer.Start();
	    PoissonImageCloning(
		background_s.get_gpu_ro(),
		target_s.get_gpu_ro(),
		mask_s.get_gpu_ro(),
		output_s.get_gpu_wo(),
		wb, hb, wt, ht, oy, ox,
		base, true, dre, bScale, start, testL
	    );
    }

    timer.Pause();

	unique_ptr<uint8_t[]> o(new uint8_t[SIZEB]);
	const float *o_cpu = output_s.get_cpu_ro();
	transform(o_cpu, o_cpu+SIZEB, o.get(), [](float f) -> uint8_t { return max(min(int(f+0.5f), 255), 0); });

    float diff = 0.0f;

    for(int i=0;i<wt;i++) {
        for(int j=0;j<ht;j++) {
            if(imgm[i*wt+j] > 127.0f && ox+i < wb && oy+j < hb) {
                int bp = ((oy+j)*wb+ox+i)*3;
                diff += (imgTest[bp] + imgTest[bp+1] + imgTest[bp+2] - o.get()[bp] - o.get()[bp+1] - o.get()[bp+2])/3;
            }
        }
    }

    file<<argv[7]<<','<<argv[8]<<','<<argv[9]<<','<<argv[10]<<','<<argv[11]<<','<<argv[12]<<','<<diff/wt/ht<<','<<timer.get_count()<<'\n';

    file.close();

	WritePPM(o.get(), wb, hb, argv[6]);
	return 0;
}
