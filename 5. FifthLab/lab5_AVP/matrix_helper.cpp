#include "matrix_helper.h"


inline unsigned int index(unsigned int height, unsigned int width, unsigned int width_size)
{
	return  height * width_size + width;
}

inline Matrix initialize_matrix(const unsigned int height, const unsigned int width)
{
	const Matrix matrix{
		new unsigned char[height*width],
		height,
		width
	};

	return matrix;
}

Matrix get_extended_matrix(Matrix old_matrix)
{
	auto new_matrix = initialize_matrix(old_matrix.height + 2, old_matrix.width + 2);

	new_matrix.matrix[0] = old_matrix.matrix[0];

	new_matrix.matrix[index(0, new_matrix.width - 1, new_matrix.width)] =
		old_matrix.matrix[index(0, old_matrix.width - 1, old_matrix.width)];

	new_matrix.matrix[index(new_matrix.height - 1, 0, new_matrix.width)] =
		old_matrix.matrix[index(old_matrix.height - 1, 0, old_matrix.width)];

	new_matrix.matrix[index(new_matrix.height - 1, new_matrix.width - 1, new_matrix.width)] =
		old_matrix.matrix[index(old_matrix.height - 1, old_matrix.width - 1, old_matrix.width)];

	for (int i = 0; i < old_matrix.height; i++)
	{
		new_matrix.matrix[index(i + 1, 0, new_matrix.width)]
			= old_matrix.matrix[index(i, 0, old_matrix.width)];
		new_matrix.matrix[index(i + 1, new_matrix.width - 1, new_matrix.width)]
			= old_matrix.matrix[index(i, old_matrix.width - 1, old_matrix.width)];
	}

	for (int j = 0; j < old_matrix.width; j++)
	{
		new_matrix.matrix[index(0, j + 1, new_matrix.width)]
			= old_matrix.matrix[index(0, j, old_matrix.width)];
		new_matrix.matrix[index(new_matrix.height - 1, j + 1, new_matrix.width)]
			= old_matrix.matrix[index(old_matrix.height - 1, j, old_matrix.width)];
	}

	for (int i = 0; i < old_matrix.height; i++)
	{
		for (int j = 0; j < old_matrix.width; j++)
		{
			new_matrix.matrix[index(i + 1, j + 1, new_matrix.width)]
				= old_matrix.matrix[index(i, j, old_matrix.width)];
		}
	}

	return new_matrix;
}

Matrix create_new_type_matrix(unsigned char* matrix_array, const unsigned int height, const unsigned int width)
{
	const Matrix matrix{
		matrix_array,
		height,
		width
	};

	return matrix;
}

Matrix HARDCORE_COMPARE(Matrix first, Matrix second)
{
	auto result_matrix = initialize_matrix(first.height, first.width);

	for(int height = 0; height < first.height; height++)
		for(int width = 0; width < first.width; width++)
		{
			if(first.matrix[index(height,width,first.width)] 
				!= second.matrix[index(height, width, second.width)])
			{
				result_matrix.matrix[index(height, width, first.width)] = 255;
			} else
			{
				result_matrix.matrix[index(height, width, first.width)] = 0;
			}
		}

	return result_matrix;
}
