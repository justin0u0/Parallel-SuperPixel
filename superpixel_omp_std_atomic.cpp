#define USE_STD_ATOMIC

#ifdef USE_STD_ATOMIC
#include <atomic>
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include "superpixel.h"
#include "superpixel_png.h"

#define RESIDUAL_ERROR_THRESHOLD 20

void clustering(Pixel* pixels, int height, int width, int s, Center* centers, int num_centers);
bool recentering(Pixel* pixels, int height, int width, int s, Center* centers, int num_centers);

int main(int argc, char** argv) {
	TimePoint start = std::chrono::steady_clock::now();

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

			++label;
		}
	}
	// std::cout << "label = " << label << std::endl;
	// std::cout << "nsx * nsy = " << nsx * nsy << std::endl;
	assert(label == nsx * nsy);

	do {
		clustering(pixels, height, width, s, centers, label);
	} while (recentering(pixels, height, width, s, centers, label));

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

	int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
	std::cout << "elapsed time = " << elapsed << " ms" << std::endl;

	return 0;
}

void clustering(Pixel* pixels, int height, int width, int s, Center* centers, int num_centers) {
	#pragma omp parallel for
	for (int k = 0; k < num_centers; ++k) {
		Center& c = centers[k];

		#pragma omp parallel for collapse(2)
		for (int i = c.x - s; i < c.x + s; ++i) {
			for (int j = c.y - s; j < c.y + s; ++j) {
				if (i < 0 || i >= height || j < 0 || j >= width) {
					continue;
				}

				int index = i * width + j;
				Pixel& p = pixels[index];

				int d1 = (c.r - p.r) * (c.r - p.r) + (c.g - p.g) * (c.g - p.g) + (c.b - p.b) * (c.b - p.b)
					+ (c.x - i) * (c.x - i) + (c.y - j) * (c.y - j);

				Center c2 = centers[p.label.load()];
				int d2 = (c2.r - p.r) * (c2.r - p.r) + (c2.g - p.g) * (c2.g - p.g) + (c2.b - p.b) * (c2.b - p.b)
					+ (c2.x - i) * (c2.x - i) + (c2.y - j) * (c2.y - j);

				if (d1 < d2) {
					p.label.store(k);
				}
			}
		}
	}
}

bool recentering(Pixel* pixels, int height, int width, int s, Center* centers, int num_centers) {
	std::atomic<int> errors(0);

	#pragma omp parallel for
	for (int k = 0; k < num_centers; ++k) {
		Center& c = centers[k];
		std::atomic<int> r(0);
		std::atomic<int> g(0);
		std::atomic<int> b(0);
		std::atomic<int> x(0);
		std::atomic<int> y(0);
		std::atomic<int> count(0);

		#pragma omp parallel for collapse(2)
		for (int i = c.x - s; i < c.x + s; ++i) {
			for (int j = c.y - s; j < c.y + s; ++j) {
				if (i < 0 || i >= height || j < 0 || j >= width) {
					continue;
				}

				int index = i * width + j;
				Pixel& p = pixels[index];
				if (p.label == k) {
					r += p.r;
					g += p.g;
					b += p.b;
					x += i;
					y += j;
					++count;
				}
			}
		}

		if (count != 0) {
			Center old = c;
			c.r = r / count;
			c.g = g / count;
			c.b = b / count;
			c.x = x / count;
			c.y = y / count;

			// compute residual error
			int error = (old.r - c.r) * (old.r - c.r) + (old.g - c.g) * (old.g - c.g) + (old.b - c.b) * (old.b - c.b)
				+ (old.x - c.x) * (old.x - c.x) + (old.y - c.y) * (old.y - c.y);
			errors += error;
		}
	}

	// std::cout << "errors = " << errors << std::endl;

	return errors > RESIDUAL_ERROR_THRESHOLD;
}