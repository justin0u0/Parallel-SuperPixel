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
	uint16_t x; // 2 bytes
	uint16_t y; // 2 bytes
	uint8_t r;  // 1 byte
	uint8_t g;  // 1 byte
	uint8_t b;  // 1 byte
	uint8_t _;  // 1 byte
};

#include <chrono>

typedef std::chrono::steady_clock::time_point TimePoint;

#endif // _SUPERPIXEL_H_
