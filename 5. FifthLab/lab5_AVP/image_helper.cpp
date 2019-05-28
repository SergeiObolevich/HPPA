#include "image_helper.h"

Image open_image(const char* image_path)
{
	unsigned char* image_array = NULL;
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int channels = 1;	

	__loadPPM(image_path, &image_array, &width, &height, &channels);

	const Image new_image{
		create_new_type_matrix(image_array,height,width),
		image_path
	};

	return new_image;
}

Image extend_image(Image image)
{
	const Image new_image{
		get_extended_matrix(image.matrix),
		"..\\images\\result\\newExtendedImage.pgm"
	};

	return new_image;
}

void save_image(Image image)
{
	__savePPM(image.path, image.matrix.matrix, image.matrix.width, image.matrix.height, 1);
}
