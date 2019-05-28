#pragma once
#include "../common/inc/helper_image.h"
#include "matrix_helper.h"

struct Image
{
	Matrix matrix;
	const char* path;
};

Image open_image(const char* image_path);
Image extend_image(Image image);
void save_image(Image image);