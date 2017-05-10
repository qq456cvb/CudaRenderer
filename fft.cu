__device__ void fft_shuffle(int thread_id, complex *vec, complex *temp, int size, int num_times) {
	int row = thread_id / size;
	int shuffle_size, shuffle_area, location, next_location;
	int odd;
	int cur = 0, next = 1;
	// need a temp matrix to record intermediate variable
	complex *arr[2];
	arr[0] = vec;
	arr[1] = temp;
	for (int i = 0; i < num_times - 1; i++) {
		shuffle_size = size / (1 << i);
		//find the calculation area
		shuffle_area = (thread_id - row * size) / shuffle_size;
		location = thread_id - row * size - shuffle_area * shuffle_size;
		// if even, "odd" = 0, if odd, "odd" = 1;
		odd = location % 2;
		// find the new index of the vector
		next_location = odd * shuffle_size / 2 + location / 2 + shuffle_area * shuffle_size + row * size; 
		arr[next][next_location].real = arr[cur][thread_id].real;
		arr[next][next_location].imag = arr[cur][thread_id].imag;
		next = cur;
		cur = !cur;
		__syncthreads();
	}
	vec[thread_id].real = arr[cur][thread_id].real;
	vec[thread_id].imag = arr[cur][thread_id].imag;
	__syncthreads();
}

__device__ void butterfly_computation(int thread_id, complex *vec, complex *temp, int size, int num_times) {
	int row = thread_id / size;
	int col = thread_id - row * size;
	int cal_size, cal_area, flag;
	int location, another_location;
	float real, imag;
	int cur = 0, next = 1;
	// need a temp matrix to record intermediate variable
	complex *arr[2];
	arr[0] = vec;
	arr[1] = temp;
	for (int i = 0; i < num_times; i++) {
		cal_size = 1 << (i + 1);
		cal_area = (thread_id - row * size) / cal_size;
		// first input element
		location = thread_id - row * size - cal_area * cal_size;
		flag = location / (cal_size / 2) * 2 - 1;
		// second input element
		another_location = location + (-1 * flag) * (cal_size / 2) + cal_area * cal_size + row * size;

		real = cosf(1.0 * location / (1 << (i + 1)) * 2 * PI);
		imag = -sinf(1.0 * location / (1 << (i + 1)) * 2 * PI);

		// butterfly computation
		arr[next][thread_id].real = (arr[cur][thread_id].real + arr[cur][another_location].real * real - arr[cur][another_location].imag * imag)* (!((flag + 1) / 2))
			+ (arr[cur][another_location].real + arr[cur][thread_id].real * real - arr[cur][thread_id].imag * imag) * (((flag + 1) / 2));
		arr[next][thread_id].imag = (arr[cur][thread_id].imag + arr[cur][another_location].real * imag + arr[cur][another_location].imag * real)* (!((flag + 1) / 2))
			+ (arr[cur][another_location].imag + arr[cur][thread_id].real * imag + arr[cur][thread_id].imag * real) * (((flag + 1) / 2));

		next = cur;
		cur = !cur;
		__syncthreads();
	}
	vec[thread_id].real = arr[cur][thread_id].real;
	vec[thread_id].imag = arr[cur][thread_id].imag;
	__syncthreads();
	/* matrix inversion */
	temp[col * size + row].real = vec[thread_id].real;
	temp[col * size + row].imag = vec[thread_id].imag;
	__syncthreads();
}

__global__ void fft2_kernel(complex *vec, complex *temp, int size, int num_times)
{
	int thread_id = blockIdx.x * threads_per_block + threadIdx.x;
	//rearrange the matrix
	fft_shuffle(thread_id, vec, temp, size, num_times);  
	// do butterfly_computation
	butterfly_computation(thread_id, vec, temp, size, num_times);
	fft_shuffle(thread_id, temp, vec, size, num_times);
	butterfly_computation(thread_id, temp, vec, size, num_times);
}