CXX = g++
CXXFLAGS = -std=c++11
LDFLAGS = -lpng
DEPS = superpixel.h superpixel_png.h
TARGETS = superpixel superpixel_omp superpixel_omp_atomic superpixel_omp_std_atomic
CUDA_TARGETS = superpixel_cuda superpixel_cuda_v1 superpixel_cuda_v2 superpixel_cuda_v3 superpixel_cuda_v4 superpixel_cuda_v5

superpixel_omp: CXXFLAGS += -fopenmp
superpixel_omp_atomic: CXXFLAGS += -fopenmp
superpixel_omp_std_atomic: CXXFLAGS += -fopenmp

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61

.PHONY: all
all: $(TARGETS) $(CUDA_TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(CUDA_TARGETS)

$(TARGETS): %: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

$(CUDA_TARGETS): %: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<
