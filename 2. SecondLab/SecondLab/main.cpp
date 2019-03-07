#include <cstdlib>
#include <malloc.h>
#include <iostream>
#include <iomanip>
#include <future>
#include <windows.h>

using namespace std;

const long long int CACHE_LINE_SIZE = 64;
const long long int CACHE_L3_SIZE = 3 * 1024 * 1024;
const long long int OFFSET = 128 * 1024 * 1024;
const long long int OFFSET_SIZE = OFFSET / sizeof(long long int);
const long long int ASSOCIATION = 20;

void initialize(long long int* array, const long long int associativity, const long long int elementsInBlock) {
	for (long long int elementIndex = 0; elementIndex < elementsInBlock; elementIndex++) {
		for (long long int blockIndex = 0; blockIndex < associativity - 1; blockIndex++) {
			array[blockIndex * OFFSET_SIZE + elementIndex] = (blockIndex + 1) * OFFSET_SIZE + elementIndex;
		}
		if (elementIndex == elementsInBlock - 1) {
			array[(associativity - 1) * OFFSET_SIZE + elementIndex] = 0;
		}
		else {
			array[(associativity - 1) * OFFSET_SIZE + elementIndex] = elementIndex + 1;
		}
	}
}

unsigned long long int readArray(long long int array[]) {
	long long int index = 0;
	const int tries = 300;
	const long long int startTime = __rdtsc();
	for (int i = 0; i < tries; i++) {
		do {
			index = array[index];
		} while (index != 0);
	}
	const long long int endTime = __rdtsc();
	return (endTime - startTime) / tries;
}

int main() {
	for (auto associativity = 1; associativity < ASSOCIATION + 1; associativity++) {
		long long int* array = static_cast<long long int*>(_aligned_malloc(OFFSET * ASSOCIATION, CACHE_LINE_SIZE));
		const long long int elementsInBlock = ceil((double)CACHE_L3_SIZE / (sizeof(long long int) * (associativity)));
		initialize(array, associativity, elementsInBlock);
		unsigned long long int time = readArray(array);
		_aligned_free(array);
		cout << associativity << "." << " " << time << " clock cycles." << endl;
	}
	system("pause");
	return 0;
}