#ifndef NN_H_
#define NN_H_

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);

typedef struct
{
	size_t rows;
	size_t cols;
	float *elements;
} Mat;

typedef struct
{
	size_t cols;
	float *elements;
} Row;

#define ROW_AT(row, col) (row).elements[col]

Mat row_as_mat(Row row);
#define row_alloc(r, cols) mat_row(mat_alloc(r, 1, cols), 0)
Row row_slice(Row row, size_t i, size_t cols);
#define row_rand(row, low, high) mat_rand(row_as_mat(row), low, high)
#define row_fill(row, x) mat_fill(row_as_mat(row), x);
#define row_print(row, name, padding) mat_print(row_as_mat(row), name, padding)
#define row_copy(dst, src) mat_copy(row_as_mat(dst), row_as_mat(src))

#define MAT_AT(m, i, j) (m).elements[(i)*(m).cols + (j)]

Mat mat_alloc(Region *r, size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Row mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_act(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_shuffle_rows(Mat m);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct
{
	size_t *arch;
	size_t arch_count;
	Mat *ws; // The amount of activations is arch_count-1
	Row *bs; // The amount of activations is arch_count-1

	// TODO: maybe remove these? It would be better to allocate them in a
	// temporary region during the actual forwarding
	Row *as;
} NN;

#define NN_INPUT(nn) (assert((nn).arch_count > 0), (nn).as[0])
#define NN_OUTPUT(nn) (assert((nn).arch_count > 0), (nn).as[(nn).arch_count-1])

NN nn_alloc(Region *r, size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
// TODO: make nn_forward signature more natural
//
// Something more like `Mat nn_forward(NN nn, Mat in)`
void nn_forward(NN nn);
float nn_cost(NN nn, Mat t);
NN nn_finite_diff(Region *r, NN nn, Mat t, float eps);
NN nn_backprop(Region *r, NN nn, Mat t);
void nn_learn(NN nn, NN g, float rate);

typedef struct
{
	size_t begin;
	float cost;
	bool finished;
} Batch;

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Mat t, float rate);

#endif // NN_H_

float actf(float x)
{
	return 1.f/(1.f+expf(-x));
}

float dactf(float y)
{
	return y*(1-y);
}

float rand_float(void)
{
	return (float)rand()/(float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
	Mat m;
	m.rows=rows;
	m.cols=cols;
	m.elements=malloc(sizeof(*m.elements)*rows*cols);
	assert(m.elements!=NULL);
	return m;
}

void mat_dot(Mat dst, Mat a, Mat b)
{
	assert(a.cols==b.rows);
	size_t n=a.cols;
	assert(dst.rows==a.rows);
	assert(dst.cols==b.cols);

	for(size_t i=0; i<dst.rows; ++i)
	{
		for(size_t j=0; j<dst.cols; ++j)
		{
			MAT_AT(dst, i, j)=0;
			for(size_t k=0; k<n; ++k)
			{
				MAT_AT(dst, i, j)+=MAT_AT(a, i, k)*MAT_AT(b, k, j);
			}
		}
	}
}

Row mat_row(Mat m, size_t row)
{
	return (Row)
	{
		.cols=m.cols,
			.elements=&MAT_AT(m, row, 0),
	};
}

void mat_copy(Mat dst, Mat src)
{
	assert(dst.rows==src.rows);
	assert(dst.cols==src.cols);
	for(size_t i=0; i<dst.rows; ++i)
	{
		for(size_t j=0; j<dst.cols; ++j)
		{
			MAT_AT(dst, i, j)=MAT_AT(src, i, j);
		}
	}
}

void mat_sum(Mat dst, Mat a)
{
	assert(dst.rows==a.rows);
	assert(dst.cols==a.cols);
	for(size_t i=0; i<dst.rows; ++i)
	{
		for(size_t j=0; j<dst.cols; ++j)
		{
			MAT_AT(dst, i, j)+=MAT_AT(a, i, j);
		}
	}
}

void mat_act(Mat m)
{
	for(size_t i=0; i<m.rows; ++i)
	{
		for(size_t j=0; j<m.cols; ++j)
		{
			MAT_AT(m, i, j)=actf(MAT_AT(m, i, j));
		}
	}
}

void mat_print(Mat m, const char *name, size_t padding)
{
	printf("%*s%s = [\n", (int)padding, "", name);
	for(size_t i=0; i<m.rows; ++i)
	{
		printf("%*s    ", (int)padding, "");
		for(size_t j=0; j<m.cols; ++j)
		{
			printf("%f ", MAT_AT(m, i, j));
		}
		printf("\n");
	}
	printf("%*s]\n", (int)padding, "");
}

void mat_fill(Mat m, float x)
{
	for(size_t i=0; i<m.rows; ++i)
	{
		for(size_t j=0; j<m.cols; ++j)
		{
			MAT_AT(m, i, j)=x;
		}
	}
}

void mat_rand(Mat m, float low, float high)
{
	for(size_t i=0; i<m.rows; ++i)
	{
		for(size_t j=0; j<m.cols; ++j)
		{
			MAT_AT(m, i, j)=rand_float()*(high-low)+low;
		}
	}
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
	assert(arch_count>0);

	NN nn;
	nn.arch=arch;
	nn.arch_count=arch_count;

	nn.ws=malloc(sizeof(*nn.ws)*(nn.arch_count-1));
	assert(nn.ws!=NULL);
	nn.bs=malloc(sizeof(*nn.bs)*(nn.arch_count-1));
	assert(nn.bs!=NULL);
	nn.as=malloc(sizeof(*nn.as)*nn.arch_count);
	assert(nn.as!=NULL);

	nn.as[0]=row_alloc(r, arch[0]);
	for(size_t i=1; i<arch_count; ++i)
	{
		nn.ws[i-1]=mat_alloc(r, nn.as[i-1].cols, arch[i]);
		nn.bs[i-1]=row_alloc(r, arch[i]);
		nn.as[i]=row_alloc(r, arch[i]);
	}

	return nn;
}

void nn_zero(NN nn)
{
	for(size_t i=0; i<nn.arch_count-1; ++i)
	{
		mat_fill(nn.ws[i], 0);
		row_fill(nn.bs[i], 0);
		row_fill(nn.as[i], 0);
	}
	row_fill(nn.as[nn.arch_count-1], 0);
}

void nn_print(NN nn, const char *name)
{
	char buf[256];
	printf("%s = [\n", name);
	for(size_t i=0; i<nn.arch_count-1; ++i)
	{
		snprintf(buf, sizeof(buf), "ws%zu", i);
		mat_print(nn.ws[i], buf, 4);
		snprintf(buf, sizeof(buf), "bs%zu", i);
		row_print(nn.bs[i], buf, 4);
	}
	printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
	for(size_t i=0; i<nn.arch_count-1; ++i)
	{
		mat_rand(nn.ws[i], low, high);
		row_rand(nn.bs[i], low, high);
	}
}

void nn_forward(NN nn)
{
	for(size_t i=0; i<nn.arch_count-1; ++i)
	{
		mat_dot(row_as_mat(nn.as[i+1]), row_as_mat(nn.as[i]), nn.ws[i]);
		mat_sum(row_as_mat(nn.as[i+1]), row_as_mat(nn.bs[i]));
		mat_act(row_as_mat(nn.as[i+1]));
	}
}

float nn_cost(NN nn, Mat t)
{
	assert(NN_INPUT(nn).cols+NN_OUTPUT(nn).cols==t.cols);
	size_t n=t.rows;

	float c=0;
	for(size_t i=0; i<n; ++i)
	{
		Row row=mat_row(t, i);
		Row x=row_slice(row, 0, NN_INPUT(nn).cols);
		Row y=row_slice(row, NN_INPUT(nn).cols, NN_OUTPUT(nn).cols);

		row_copy(NN_INPUT(nn), x);

		nn_forward(nn);

		for(size_t j=0; j<y.cols; ++j)
		{
			float d=NN_OUTPUT(nn).elements[j]-y.elements[j];
			c+=d*d;
		}
	}

	return c/n;
}

NN nn_backprop(NN nn, Mat t)
{
	size_t n=t.rows;
	assert(NN_INPUT(nn).cols+NN_OUTPUT(nn).cols==t.cols);

	NN g=nn_alloc(r, nn.arch, nn.arch_count);
	nn_zero(g);

	// i - current sample
	// l - current layer
	// j - current activation
	// k - previous activation

	for(size_t i=0; i<n; ++i)
	{
		Row row=mat_row(t, i);
		Row in=row_slice(row, 0, NN_INPUT(nn).cols);
		Row out=row_slice(row, NN_INPUT(nn).cols, NN_OUTPUT(nn).cols);

		row_copy(NN_INPUT(nn), in);
		nn_forward(nn);

		for(size_t j=0; j<nn.arch_count; ++j)
		{
			row_fill(g.as[j], 0);
		}

		for(size_t j=0; j<out.cols; ++j)
		{
#ifdef NN_BACKPROP_TRADITIONAL
			NN_OUTPUT(g).elements[j]=2.0f*(NN_OUTPUT(nn).elements[j]-out.elements[j);
#else
			NN_OUTPUT(g).elements[j]=NN_OUTPUT(nn).elements[j]-out.elements[j];
#endif // NN_BACKPROP_TRADITIONAL
		}

#ifdef NN_BACKPROP_TRADITIONAL
		float s=1;
#else
		float s=2;
#endif // NN_BACKPROP_TRADITIONAL

		for(size_t l=nn.arch_count-1; l>0; --l)
		{
			for(size_t j=0; j<nn.as[l].cols; ++j)
			{
				float a=nn.as[l].elements[j];
				float da=g.as[l].elements[j];
				float qa=dactf(a);

				g.bs[l-1].elements[j]+=s*da*qa;

				for(size_t k=0; k<nn.as[l-1].cols; ++k)
				{
					float pa=nn.as[l-1].elements[k];
					float w=MAT_AT(nn.ws[l-1], k, j);

					MAT_AT(g.ws[l-1], k, j)+=s*da*qa*pa;

					g.as[l-1].elements[k]+=s*da*qa*w;
				}
			}
		}
	}

	for(size_t i=0; i<g.arch_count-1; ++i)
	{
		for(size_t j=0; j<g.ws[i].rows; ++j)
		{
			for(size_t k=0; k<g.ws[i].cols; ++k)
			{
				MAT_AT(g.ws[i], j, k)/=n;
			}
		}
		for(size_t k=0; k<g.bs[i].cols; ++k)
		{
			g.bs[i].elements[k]/=n;
		}
	}

	return g;
}

NN nn_finite_diff(NN nn, Mat t, float eps)
{
	float saved;
	float c=nn_cost(nn, t);

	NN g=nn_alloc(r, nn.arch, nn.arch_count);

	for(size_t i=0; i<nn.arch_count-1; ++i)
	{
		for(size_t j=0; j<nn.ws[i].rows; ++j)
		{
			for(size_t k=0; k<nn.ws[i].cols; ++k)
			{
				saved=MAT_AT(nn.ws[i], j, k);
				MAT_AT(nn.ws[i], j, k)+=eps;
				MAT_AT(g.ws[i], j, k)=(nn_cost(nn, t)-c)/eps;
				MAT_AT(nn.ws[i], j, k)=saved;
			}
		}

		for(size_t k=0; k<nn.bs[i].cols; ++k)
		{
			saved=nn.bs[i].elements[k];
			nn.bs[i].elements[k]+=eps;
			g.bs[i].elements[k]=(nn_cost(nn, t)-c)/eps;
			nn.bs[i].elements[k]=saved;
		}
	}

	return g;
}

void nn_learn(NN nn, NN g, float rate)
{
	for(size_t i=0; i<nn.arch_count-1; ++i)
	{
		for(size_t j=0; j<nn.ws[i].rows; ++j)
		{
			for(size_t k=0; k<nn.ws[i].cols; ++k)
			{
				MAT_AT(nn.ws[i], j, k)-=rate*MAT_AT(g.ws[i], j, k);
			}
		}

		for(size_t k=0; k<nn.bs[i].cols; ++k)
		{
			ROW_AT(nn.bs[i], k)-=rate*ROW_AT(g.bs[i], k);
		}
	}
}

void mat_shuffle_rows(Mat m)
{
	for(size_t i=0; i<m.rows; ++i)
	{
		size_t j=i+rand()%(m.rows-i);
		if(i!=j)
		{
			for(size_t k=0; k<m.cols; ++k)
			{
				float t=MAT_AT(m, i, k);
				MAT_AT(m, i, k)=MAT_AT(m, j, k);
				MAT_AT(m, j, k)=t;
			}
		}
	}
}

void batch_process(Batch *b, size_t batch_size, NN nn, Mat t, float rate)
{
	if(b->finished)
	{
		b->finished=false;
		b->begin=0;
		b->cost=0;
	}

	size_t size=batch_size;
	if(b->begin+batch_size>=t.rows)
	{
		size=t.rows-b->begin;
	}

	// TODO: introduce similar to row_slice operation but for Mat that will give you subsequence of rows
	Mat batch_t={
		.rows=size,
		.cols=t.cols,
		.elements=&MAT_AT(t, b->begin, 0),
	};

	NN g=nn_backprop(r, nn, batch_t);
	nn_learn(nn, g, rate);
	b->cost+=nn_cost(nn, batch_t);
	b->begin+=batch_size;

	if(b->begin>=t.rows)
	{
		size_t batch_count=(t.rows+batch_size-1)/batch_size;
		b->cost/=batch_count;
		b->finished=true;
	}
}

Mat row_as_mat(Row row)
{
	return (Mat)
	{
		.rows=1,
			.cols=row.cols,
			.elements=row.elements,
	};
}

Row row_slice(Row row, size_t i, size_t cols)
{
	assert(i<row.cols);
	assert(i+cols<=row.cols);

	return (Row) { .cols=cols, .elements=&row.elements[i], };
}

#endif // NN_IMPLEMENTATION
