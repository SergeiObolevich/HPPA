#pragma once
#include "image_helper.h"
#include "matrix_helper.h"

struct Task
{
	Image image;
	Matrix work_matrix;
	int filter[3][3];
	int division_coef;
};

struct Result
{
	Matrix result;
	float time;
};

Task create_task(Image extended_image);