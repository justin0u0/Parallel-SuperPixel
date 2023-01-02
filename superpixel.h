#ifndef _SUPERPIXEL_H_
#define _SUPERPIXEL_H_

#include <stdint.h>

struct Pixel {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint16_t label;
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
