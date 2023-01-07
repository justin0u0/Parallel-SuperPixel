/**
 * AtomicBlock -> AtomicThread
*/
#include <cassert>
#include <cmath>
#include <iostream>
#include "superpixel.h"
#include "superpixel_png.h"

#define RESIDUAL_ERROR_THRESHOLD 20

__global__ void clustering(Pixel* pixels, int height, int width, int s, Center* centers) {
	int k = blockIdx.z;
	Center& c = centers[k];

	int i = threadIdx.x + blockIdx.x * blockDim.x + c.y - s;
	int j = threadIdx.y + blockIdx.y * blockDim.y + c.x - s;
	if (i < 0 || i >= width || j < 0 || j >= height) {
		return;
	}

	int index = j * width + i;
	Pixel& p = pixels[index];
	Center& c2 = centers[p.label];
	int d1 = (c.r - p.r) * (c.r - p.r) + (c.g - p.g) * (c.g - p.g) + (c.b - p.b) * (c.b - p.b)
		+ (c.x - j) * (c.x - j) + (c.y - i) * (c.y - i);
	int d2 = (c2.r - p.r) * (c2.r - p.r) + (c2.g - p.g) * (c2.g - p.g) + (c2.b - p.b) * (c2.b - p.b)
		+ (c2.x - j) * (c2.x - j) + (c2.y - i) * (c2.y - i);
	if (d1 < d2) {
		p.label = k;
	}
}

__global__ void centering_phase1(Pixel* pixels, int height, int width, int s, Center* centers, int* all) {
	int k = blockIdx.x;
	Center& c = centers[k];

	int r = 0;
	int g = 0;
	int b = 0;
	int x = 0;
	int y = 0;
	int count = 0;

	for (int i = threadIdx.x + c.y - s; i < min(c.y + s, width); i += blockDim.x) {
		for (int j = threadIdx.y + c.x - s; j < min(c.x + s, height); j += blockDim.y) {
			if (i < 0 || j < 0) {
				continue;
			}

			int index = j * width + i;
			Pixel& p = pixels[index];
			if (p.label == k) {
				r += p.r;
				g += p.g;
				b += p.b;
				x += j;
				y += i;
				++count;
			}
		}
	}

	int* base = all + k * 6;
	atomicAdd(base, r);
	atomicAdd(base + 1, g);
	atomicAdd(base + 2, b);
	atomicAdd(base + 3, x);
	atomicAdd(base + 4, y);
	atomicAdd(base + 5, count);
}

__global__ void centering_phase2(Center* centers, int* all, int* errors, int num_centers) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if (k >= num_centers) {
		return;
	}

	int* base = all + k * 6;
	int count = *(base + 5);
	if (count == 0) {
		return;
	}

	Center& c = centers[k];
	Center old = c;
	c.r = *base / count;
	c.g = *(base + 1) / count;
	c.b = *(base + 2) / count;
	c.x = *(base + 3) / count;
	c.y = *(base + 4) / count;

	int error = (c.r - old.r) * (c.r - old.r) + (c.g - old.g) * (c.g - old.g) + (c.b - old.b) * (c.b - old.b)
		+ (c.x - old.x) * (c.x - old.x) + (c.y - old.y) * (c.y - old.y);
	atomicAdd_block(errors, error);
}

int main(int argc, char** argv) {
	const char* infile = argv[1];
	const char* outfile = argv[2];

	// number of superpixels
	const int k = atoi(argv[3]);

	unsigned char* image;
	unsigned int height;
	unsigned int width;
	unsigned int channels;

	int err = read_png(infile, &image, height, width, channels);
	if (err != 0) {
		printf("fail to read input png file\n");
		exit(err);
	}

	int n = height * width;
	// std::cout << "height = " << height << ", width = " << width << std::endl;

	Pixel* pixels = (Pixel*)malloc(n * sizeof(Pixel));
	for (int i = 0; i < n; ++i) {
		pixels[i].r = image[i * channels];
		pixels[i].g = image[i * channels + 1];
		pixels[i].b = image[i * channels + 2];
		pixels[i].label = 0;
	}

	// size of superpixel (s * s)
	int s = sqrt(n / k);
	// number of superpixels in x and y direction
	int nsx = (height - s / 2 - 1) / s + 1;
	int nsy = (width - s / 2 - 1) / s + 1;

	// std::cout << "s = " << s << std::endl;
	// std::cout << "nsx = " << nsx << ", nsy = " << nsy << std::endl;

	// initialize superpixels centers
	Center* centers = (Center*)malloc(nsx * nsy * sizeof(Center));
	int label = 0;
	for (int i = s / 2; i < height; i += s) {
		for (int j = s / 2; j < width; j += s) {
			int index = i * width + j;
			Center& c = centers[label];
			c.x = i;
			c.y = j;
			c.r = pixels[index].r;
			c.g = pixels[index].g;
			c.b = pixels[index].b;
			c.label = label;

			++label;
		}
	}
	// std::cout << "label = " << label << std::endl;
	// std::cout << "nsx * nsy = " << nsx * nsy << std::endl;
	assert(label == nsx * nsy);

	Pixel* dpixels;
	Center* dcenters;

	cudaMalloc((void**)&dpixels, n * sizeof(Pixel));
	cudaMalloc((void**)&dcenters, nsx * nsy * sizeof(Center));

	cudaMemcpy(dpixels, pixels, n * sizeof(Pixel), cudaMemcpyHostToDevice);
	cudaMemcpy(dcenters, centers, nsx * nsy * sizeof(Center), cudaMemcpyHostToDevice);

	int* dall; // r, g, b, x, y, count
	cudaMalloc((void**)&dall, 6 * label * sizeof(int));

	int* derrors;
	cudaMalloc((void**)&derrors, sizeof(int));

	dim3 threads_per_block(32, 32);
	dim3 num_blocks((2 * s - 1) / threads_per_block.x + 1, (2 * s - 1) / threads_per_block.y + 1, label);

	int errors = 1e9;
	while (errors > RESIDUAL_ERROR_THRESHOLD) {
		clustering<<<num_blocks, threads_per_block>>>(dpixels, height, width, s, dcenters);

		cudaMemset(dall, 0, 6 * label * sizeof(int));
		cudaMemset(derrors, 0, sizeof(int));

		centering_phase1<<<label, threads_per_block>>>(dpixels, height, width, s, dcenters, dall);
		centering_phase2<<<(label - 1) / 1024 + 1, 1024>>>(dcenters, dall, derrors, label);

		cudaMemcpy(&errors, derrors, sizeof(int), cudaMemcpyDeviceToHost);

		// std::cout << "errors = " << errors << std::endl;
	}

	cudaMemcpy(pixels, dpixels, n * sizeof(Pixel), cudaMemcpyDeviceToHost);
	cudaMemcpy(centers, dcenters, nsx * nsy * sizeof(Center), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; ++i) {
		int index = pixels[i].label;
		image[i * channels] = centers[index].r;
		image[i * channels + 1] = centers[index].g;
		image[i * channels + 2] = centers[index].b;
	}

	err = write_png(outfile, image, height, width, channels);
	if (err != 0) {
		printf("fail to write output png file\n");
		exit(err);
	}

	free(pixels);
	free(centers);
	free(image);

	return 0;
}
