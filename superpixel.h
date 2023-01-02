#ifndef _SUPERPIXEL_H_
#define _SUPERPIXEL_H_

#ifdef USE_STD_ATOMIC
#include <atomic>
#endif
#include <cstdint>

struct Pixel {
	uint8_t r;
	uint8_t g;
	uint8_t b;
#ifdef USE_STD_ATOMIC
	std::atomic<uint16_t> label;
#else
	uint16_t label;
#endif
};

// SuperPixel cluster center
struct Center {
	uint16_t x;
	uint16_t y;
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint16_t label;
};

#endif // _SUPERPIXEL_H_
