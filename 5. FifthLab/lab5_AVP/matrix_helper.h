#pragma once

struct Matrix
{
	unsigned char* matrix;
	unsigned int height;
	unsigned int width;
};

inline unsigned int index(unsigned int height, unsigned int width, unsigned int width_size);
inline Matrix initialize_matrix(const unsigned int height, const unsigned int width);
Matrix get_extended_matrix(Matrix old_matrix);
Matrix create_new_type_matrix(unsigned char* matrix_array, const unsigned int height, const unsigned int width);

Matrix HARDCORE_COMPARE(Matrix first, Matrix second);