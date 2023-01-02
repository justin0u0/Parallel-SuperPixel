CXX = g++
LDFLAGS = -lpng
TARGETS = superpixel superpixel_omp

superpixel_omp: CXXFLAGS += -fopenmp -std=c++11
# superpixel_omp2: CXXFLAGS += -fopenmp -std=c++11

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

superpixel_cuda: superpixel_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<
