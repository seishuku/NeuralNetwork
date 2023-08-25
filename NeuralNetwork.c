#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>

pthread_t DDrawThread;
void *CreateDDrawWindow(void *Arg);
extern bool Done;
extern int Width, Height;

typedef struct
{
	uint32_t Width, Height, Depth;
	uint8_t *Data;
} VkuImage_t;

bool TGA_Load(const char *Filename, VkuImage_t *Image);
bool TGA_Write(const char *filename, VkuImage_t *Image, bool rle);

#define INPUT_SIZE 2
#define HIDDEN_SIZE 28
#define OUTPUT_SIZE 3
#define LEARNING_RATE 0.1f

float randfloat(float low, float high)
{
	return ((float)rand()/(float)RAND_MAX)*(high-low)+low;
}

// Sigmoid activation function
float sigmoid(float x)
{
	return 1.0f/(1.0f+expf(-x));
}

// Derivative of sigmoid function
float sigmoid_derivative(float x)
{
	return x*(1.0f-x);
}

// Forward propagation
void forward_propagation(float *input,
						 float *input_hidden_weights, float *input_hidden_biases, float *hidden_output,
						 float *hidden_hidden_weights, float *hidden_hidden_biases, float *hidden2_output,
						 float *hidden_output_weights, float *hidden_output_biases, float *output)
{
	float weighted_sum;

	for(int i=0;i<HIDDEN_SIZE;i++)
	{
		weighted_sum=input_hidden_biases[i];

		for(int j=0;j<INPUT_SIZE;j++)
			weighted_sum+=input[j]*input_hidden_weights[i*INPUT_SIZE+j];

		hidden_output[i]=sigmoid(weighted_sum);
	}

	for(int i=0;i<HIDDEN_SIZE;i++)
	{
		weighted_sum=hidden_hidden_biases[i];

		for(int j=0;j<HIDDEN_SIZE;j++)
			weighted_sum+=hidden_output[j]*hidden_hidden_weights[i*HIDDEN_SIZE+j];

		hidden2_output[i]=sigmoid(weighted_sum);
	}

	for(int i=0;i<OUTPUT_SIZE;i++)
	{
		weighted_sum=hidden_output_biases[i];

		for(int j=0;j<HIDDEN_SIZE;j++)
			weighted_sum+=hidden2_output[j]*hidden_output_weights[i*HIDDEN_SIZE+j];

		output[i]=sigmoid(weighted_sum);
	}
}

// Back propagation
float backpropagation(float *input, float *target,
					  float *input_hidden_weights, float *input_hidden_biases, float *hidden_output,
					  float *hidden_hidden_weights, float *hidden_hidden_biases, float *hidden2_output,
					  float *hidden_output_weights, float *hidden_output_biases, float *output)
{
	float output_errors[OUTPUT_SIZE], output_deltas[OUTPUT_SIZE];
	float total_error=0.0f;

	// output error and delta
	for(int i=0;i<OUTPUT_SIZE;i++)
	{
		output_errors[i]=(target[i]-output[i]);
		output_deltas[i]=output_errors[i]*sigmoid_derivative(output[i]);

		total_error+=output_errors[i]*output_errors[i];
	}

	// Compute hidden layer errors and deltas
	float hidden2_errors[HIDDEN_SIZE], hidden2_deltas[HIDDEN_SIZE];

	for(int i=0;i<HIDDEN_SIZE;i++)
	{
		hidden2_errors[i]=0.0f;

		for(int j=0;j<OUTPUT_SIZE;j++)
			hidden2_errors[i]+=output_deltas[j]*hidden_output_weights[j*HIDDEN_SIZE+i];

		hidden2_deltas[i]=hidden2_errors[i]*sigmoid_derivative(hidden2_output[i]);
	}

	float hidden_errors[HIDDEN_SIZE], hidden_deltas[HIDDEN_SIZE];

	for(int i=0;i<HIDDEN_SIZE;i++)
	{
		hidden_errors[i]=0.0f;

		for(int j=0;j<OUTPUT_SIZE;j++)
			hidden_errors[i]+=hidden2_deltas[j]*hidden_hidden_weights[j*HIDDEN_SIZE+i];

		hidden_deltas[i]=hidden_errors[i]*sigmoid_derivative(hidden_output[i]);
	}

	// hidden->output weights and biases
	for(int i=0;i<HIDDEN_SIZE;i++)
	{
		for(int j=0;j<OUTPUT_SIZE;j++)
			hidden_output_weights[j*HIDDEN_SIZE+i]+=LEARNING_RATE*output_deltas[j]*hidden2_output[i];
	}

	for(int i=0;i<OUTPUT_SIZE;i++)
		hidden_output_biases[i]+=LEARNING_RATE*output_deltas[i];

	// hidden->hidden weights and biases
	for(int i=0;i<HIDDEN_SIZE;i++)
	{
		for(int j=0;j<HIDDEN_SIZE;j++)
			hidden_hidden_weights[j*HIDDEN_SIZE+i]+=LEARNING_RATE*hidden2_deltas[j]*hidden_output[i];
	}

	for(int i=0;i<HIDDEN_SIZE;i++)
		hidden_hidden_biases[i]+=LEARNING_RATE*hidden2_deltas[i];

	// input->hidden weights and biases
	for(int i=0;i<INPUT_SIZE;i++)
	{
		for(int j=0;j<HIDDEN_SIZE;j++)
			input_hidden_weights[j*INPUT_SIZE+i]+=LEARNING_RATE*hidden_deltas[j]*input[i];
	}

	for(int i=0;i<HIDDEN_SIZE;i++)
		input_hidden_biases[i]+=LEARNING_RATE*hidden_deltas[i];

	// return normalized total error
	return total_error/OUTPUT_SIZE;
}

float normalize_output(float x, float min_val, float max_val)
{
	return (x-min_val)/(max_val-min_val);
}

float *input_hidden_weights, *input_hidden_biases;
float *hidden_hidden_weights, *hidden_hidden_biases;
float *hidden_output_weights, *hidden_output_biases;

VkuImage_t InputImage, OutputImage;

float *copy_input_hidden_weights, *copy_input_hidden_biases;
float *copy_hidden_hidden_weights, *copy_hidden_hidden_biases;
float *copy_hidden_output_weights, *copy_hidden_output_biases;

// Training function
void train_neural_network(float *inputs, float *targets, int num_samples)
{
	float rand_value=0.7f;

	// input -> hidden1
	input_hidden_weights=(float *)malloc(INPUT_SIZE*HIDDEN_SIZE*sizeof(float));

	if(input_hidden_weights==NULL)
		return;

	for(int i=0;i<INPUT_SIZE*HIDDEN_SIZE;i++)
		input_hidden_weights[i]=randfloat(-rand_value, rand_value);

	input_hidden_biases=(float *)malloc(HIDDEN_SIZE*sizeof(float));

	if(input_hidden_biases==NULL)
		return;

	for(int i=0;i<HIDDEN_SIZE;i++)
		input_hidden_biases[i]=randfloat(-rand_value, rand_value);

	// hidden -> hidden
	hidden_hidden_weights=(float *)malloc(HIDDEN_SIZE*HIDDEN_SIZE*sizeof(float));

	if(hidden_hidden_weights==NULL)
		return;

	for(int i=0;i<HIDDEN_SIZE*HIDDEN_SIZE;i++)
		hidden_hidden_weights[i]=randfloat(-rand_value, rand_value);

	hidden_hidden_biases=(float *)malloc(HIDDEN_SIZE*sizeof(float));

	if(hidden_hidden_biases==NULL)
		return;

	for(int i=0;i<HIDDEN_SIZE;i++)
		hidden_hidden_biases[i]=randfloat(-rand_value, rand_value);

	// hidden2 -> output
	hidden_output_weights=(float *)malloc(HIDDEN_SIZE*OUTPUT_SIZE*sizeof(float));

	if(hidden_output_weights==NULL)
		return;

	for(int i=0;i<HIDDEN_SIZE*OUTPUT_SIZE;i++)
		hidden_output_weights[i]=randfloat(-rand_value, rand_value);

	hidden_output_biases=(float *)malloc(OUTPUT_SIZE*sizeof(float));

	if(hidden_output_biases==NULL)
		return;

	for(int i=0;i<OUTPUT_SIZE;i++)
		hidden_output_biases[i]=randfloat(-rand_value, rand_value);

	// Copy of data for double buffering (to render live output)
	// input -> hidden1
	copy_input_hidden_weights=(float *)malloc(INPUT_SIZE*HIDDEN_SIZE*sizeof(float));

	if(copy_input_hidden_weights==NULL)
		return;

	copy_input_hidden_biases=(float *)malloc(HIDDEN_SIZE*sizeof(float));

	if(copy_input_hidden_biases==NULL)
		return;

	// hidden -> hidden
	copy_hidden_hidden_weights=(float *)malloc(HIDDEN_SIZE*HIDDEN_SIZE*sizeof(float));

	if(copy_hidden_hidden_weights==NULL)
		return;

	copy_hidden_hidden_biases=(float *)malloc(HIDDEN_SIZE*sizeof(float));

	if(copy_hidden_hidden_biases==NULL)
		return;

	// hidden2 -> output
	copy_hidden_output_weights=(float *)malloc(HIDDEN_SIZE*OUTPUT_SIZE*sizeof(float));

	if(copy_hidden_output_weights==NULL)
		return;

	copy_hidden_output_biases=(float *)malloc(OUTPUT_SIZE*sizeof(float));

	if(copy_hidden_output_biases==NULL)
		return;

	// Train the data until keypress
	float hidden_output[HIDDEN_SIZE], hidden2_output[HIDDEN_SIZE], output[OUTPUT_SIZE];

	for(int epoch=0;;epoch++)
	{
		float total_error=0.0f;

		for(int sample=0;sample<num_samples;sample++)
		{
			forward_propagation(&inputs[INPUT_SIZE*sample],
								input_hidden_weights, input_hidden_biases, hidden_output,
								hidden_hidden_weights, hidden_hidden_biases, hidden2_output,
								hidden_output_weights, hidden_output_biases, output);

			total_error+=backpropagation(&inputs[INPUT_SIZE*sample], &targets[OUTPUT_SIZE*sample],
										 input_hidden_weights, input_hidden_biases, hidden_output,
										 hidden_hidden_weights, hidden_hidden_biases, hidden2_output,
										 hidden_output_weights, hidden_output_biases, output);

		}

		for(int i=0;i<OUTPUT_SIZE;i++)
			output[i]=normalize_output(output[i], 0.0f, 1.0f);

		total_error/=num_samples;

		printf("Epoch %d, Loss: %f\r", epoch+1, total_error);

		memcpy(copy_input_hidden_weights, input_hidden_weights, sizeof(float)*INPUT_SIZE*HIDDEN_SIZE);
		memcpy(copy_input_hidden_biases, input_hidden_biases, sizeof(float)*HIDDEN_SIZE);

		memcpy(copy_hidden_hidden_weights, hidden_hidden_weights, sizeof(float)*HIDDEN_SIZE*HIDDEN_SIZE);
		memcpy(copy_hidden_hidden_biases, hidden_hidden_biases, sizeof(float)*HIDDEN_SIZE);

		memcpy(copy_hidden_output_weights, hidden_output_weights, sizeof(float)*HIDDEN_SIZE*OUTPUT_SIZE);
		memcpy(copy_hidden_output_biases, hidden_output_biases, sizeof(float)*OUTPUT_SIZE);

		if(_kbhit())
				break;
	}

	putc('\n', stdout);
}

int main()
{
	TGA_Load("testcolor3.tga", &InputImage);

	int num_samples=InputImage.Width*InputImage.Height;

	float *inputs=(float *)malloc(sizeof(float *)*INPUT_SIZE*num_samples);

	if(inputs==NULL)
		return -1;

	for(uint32_t y=0;y<InputImage.Height;y++)
	{
		float dy=(float)y/(float)(InputImage.Height+1);

		for(uint32_t x=0;x<InputImage.Width;x++)
		{
			float dx=(float)x/(float)(InputImage.Width+1);
			size_t Index=y*InputImage.Width+x;

			inputs[INPUT_SIZE*Index+0]=dx;
			inputs[INPUT_SIZE*Index+1]=dy;
		}
	}

	float *targets=malloc(sizeof(float *)*OUTPUT_SIZE*num_samples);

	if(targets==NULL)
		return -1;

	for(uint32_t y=0;y<InputImage.Height;y++)
	{
		for(uint32_t x=0;x<InputImage.Width;x++)
		{
			size_t TargetIndex=OUTPUT_SIZE*(y*InputImage.Width+x);
			size_t ImageIndex=3*(y*InputImage.Width+x);

			targets[TargetIndex+0]=(float)InputImage.Data[ImageIndex+0]/255.0f;
			targets[TargetIndex+1]=(float)InputImage.Data[ImageIndex+1]/255.0f;
			targets[TargetIndex+2]=(float)InputImage.Data[ImageIndex+2]/255.0f;
		}
	}

	free(InputImage.Data);

	Width=InputImage.Width*4;
	Height=InputImage.Height*4;
	pthread_create(&DDrawThread, NULL, CreateDDrawWindow, NULL);

	train_neural_network(inputs, targets, num_samples);

	// Using the trained neural network to reproduce the original data, but scaled 8x
	OutputImage.Width=InputImage.Width*8;
	OutputImage.Height=InputImage.Height*8;
	OutputImage.Depth=InputImage.Depth;

	num_samples=OutputImage.Width*OutputImage.Height;

	OutputImage.Data=malloc(3*num_samples);

	float hidden_output[HIDDEN_SIZE], hidden2_output[HIDDEN_SIZE];

	for(uint32_t y=0;y<OutputImage.Height;y++)
	{
		float dy=(float)y/(float)OutputImage.Height;

		for(uint32_t x=0;x<OutputImage.Width;x++)
		{
			float dx=(float)x/(float)OutputImage.Width;

			float input[INPUT_SIZE]={ dx, dy };
			float output[OUTPUT_SIZE];

			forward_propagation(input,
								input_hidden_weights, input_hidden_biases, hidden_output,
								hidden_hidden_weights, hidden_hidden_biases, hidden2_output,
								hidden_output_weights, hidden_output_biases, output);

			size_t Index=3*(y*OutputImage.Width+x);

			OutputImage.Data[Index+0]=(uint8_t)(output[0]*255.0f);
			OutputImage.Data[Index+1]=(uint8_t)(output[1]*255.0f);
			OutputImage.Data[Index+2]=(uint8_t)(output[2]*255.0f);
		}
	}

	TGA_Write("output.tga", &OutputImage, false);

	Done=true;
	pthread_join(DDrawThread, NULL);

	return 0;
}
