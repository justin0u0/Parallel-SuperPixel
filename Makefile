CXX = g++
LDFLAGS = -lpng
TARGETS = superpixel superpixel_omp superpixel_omp_atomic superpixel_omp_std_atomic superpixel_cuda

superpixel_omp: CXXFLAGS += -fopenmp -std=c++11
superpixel_omp_atomic: CXXFLAGS += -fopenmp -std=c++11
superpixel_omp_std_atomic: CXXFLAGS += -fopenmp -std=c++11
superpixel_cuda: CXXFLAGS += -std=c++11

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

superpixel_cuda: superpixel_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<
