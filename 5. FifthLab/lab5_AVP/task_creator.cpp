#include "task_creator.h"

Task create_task(Image extended_image)
{
	const Task task{
		extended_image,
		initialize_matrix(extended_image.matrix.height - 2, extended_image.matrix.width - 2),
		{ { 1,-2,1 },{ -2,5,-2 },{ 1,-2,1 } },
		1
	};

	return task;
}