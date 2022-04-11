#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<sys/time.h>
#include<ctime>

//#include<nmmintrin.h>  // SSE 4,2
//#include<immintrin.h>  // AVX

#include<arm_neon.h>  // neon


using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

void reset(float** a, int n);
void c_reset(float** a, int n);
void show(float** a, int n);
void GE(float** a, int n);
void C_GE(float** a, int n);
void neon_GE(float** a, int n);
void neon_U_GE(float** a, int n);
void neon_U_GE_Loop1(float** a, int n);
void neon_U_GE_Loop2(float** a, int n);
void neon_U_GE_Opt1(float** a, int n);  // 流水线优化
void neon_U_GE_fms(float** a, int n);  // 使用fmsq指令，即融合乘减指令

int n = 1000;
int lim = 1;
ofstream out("output.txt");
//ifstream in("input.txt");


void GE(float** a, int n) {  // 标准的高斯消去算法, Gauss Elimination缩写
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j ++)
		{
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE(float** a, int n) {  // 高斯消去算法的Cache优化版本
	//__m128 va;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void neon_GE(float** a, int n) {  // 使用neon进行SIMD优化的高斯消去算法
	float32x4_t va, vt, vaik, vakj, vaij, vx;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(a[k][k]);  // 加载四个重复值到vt中
		t1 = a[k][k];
		int j = 0;
		for (j = k + 1; j + 4 < n; j += 4)
		{
			if (j % 4 != 0) {
				a[k][j] = a[k][j] / t1;
				j -= 3;
				continue;
			}
			//cout<<&a[k][j]<<endl;
			va = vld1q_f32(&a[k][j]);
			va = vdivq_f32(va, vt);
			vst1q_f32((float32_t*)&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // 善后
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				if (j % 4 != 0) {
					a[i][j] -= t2 * a[k][j];
					j -= 3;
					continue;
				}
				vakj = vld1q_f32(&a[k][j]);
				vaij = vld1q_f32(&a[i][j]);
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32((float32_t*)&a[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void neon_U_GE(float** a, int n) {  // 使用neon进行SIMD优化的高斯消去算法，未对齐
	float32x4_t va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(a[k][k]);  // 加载四个重复值到vt中
		int j = 0;
		for (j = k + 1; j + 4 < n; j += 4)
		{
			va = vld1q_f32(&a[k][j]);
			va = vdivq_f32(va, vt);
			vst1q_f32((float32_t*)&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // 善后
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = vld1q_f32(&a[k][j]);
				vaij = vld1q_f32(&a[i][j]);
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				//vaij = vfmsq_f32(vaij,vakj,vaik);
				vst1q_f32((float32_t*)&a[i][j], vaij);
				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void neon_U_GE_fms(float** a, int n) {  // 使用neon进行SIMD优化的高斯消去算法，未对齐，使用fmsq
	float32x4_t va, vt, vaik, vakj, vaij;
	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(a[k][k]);  // 加载四个重复值到vt中
		int j = 0;
		for (j = k + 1; j + 4 < n; j += 4)
		{
			va = vld1q_f32(&a[k][j]);
			va = vdivq_f32(va, vt);
			vst1q_f32((float32_t*)&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // 善后
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = vld1q_f32(&a[k][j]);
				vaij = vld1q_f32(&a[i][j]);
				vaij = vfmsq_f32(vaij,vakj,vaik);
				vst1q_f32((float32_t*)&a[i][j], vaij);
				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void neon_U_GE_Loop1(float** a, int n) {  // 使用neon进行SIMD优化的高斯消去算法，未对齐，只优化第一个循环
	float32x4_t va, vt;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(a[k][k]);  // 加载四个重复值到vt中
		int j = 0;
		for (j = k + 1; j + 4 < n; j += 4)
		{
			va = vld1q_f32(&a[k][j]);
			va = vdivq_f32(va, vt);
			vst1q_f32((float32_t*)&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // 善后
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (j = k + 1; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void neon_U_GE_Loop2(float** a, int n) {  // 使用neon进行SIMD优化的高斯消去算法，未对齐，只优化第二个循环
	float32x4_t vaik, vakj, vaij, vx;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];  // 加载四个重复值到vt中
		int j = 0;
		for (j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = vld1q_f32(&a[k][j]);
				vaij = vld1q_f32(&a[i][j]);
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32((float32_t*)&a[i][j], vaij);
				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void neon_U_GE_Opt1(float** a, int n) {  // 使用neon进行SIMD优化的高斯消去算法，未对齐，尝试采用流水线优化
	float32x4_t va1, va2, vt1, vt2, vaik1, vaik2, vakj1, vakj2, vaij1, vaij2, vx1, vx2;
	for (int k = 0; k < n; k++)
	{
		vt1 = vmovq_n_f32(a[k][k]);  // 加载四个重复值到vt1中
		vt2 = vmovq_n_f32(a[k][k]);  // 加载四个重复值到vt2中
		int j = 0;
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va1 = vld1q_f32(&a[k][j]);
			va2 = vld1q_f32(&a[k][j] + 4);
			va1 = vdivq_f32(va1, vt1);
			va2 = vdivq_f32(va2, vt2);
			vst1q_f32((float32_t*)&a[k][j], va1);
			vst1q_f32((float32_t*)(&a[k][j] + 4), va2);  // 流水线策略
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // 善后
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik1 = vmovq_n_f32(a[i][k]);
			vaik2 = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj1 = vld1q_f32(&a[k][j]);
				vakj2 = vld1q_f32(&a[k][j] + 4);
				vaij1 = vld1q_f32(&a[i][j]);
				vaij2 = vld1q_f32(&a[i][j] + 4);

				vx1 = vmulq_f32(vakj1, vaik1);
				vx2 = vmulq_f32(vakj2, vaik2);

				vaij1 = vsubq_f32(vaij1, vx1);
				vaij2 = vsubq_f32(vaij2, vx2);

				vst1q_f32((float32_t*)&a[i][j], vaij1);
				vst1q_f32((float32_t*)(&a[i][j] + 4), vaij2);  // 流水线策略

				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

float** generate(int n) {
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
	}
	reset(m, n);
	return m;
}

float** aligned_generate(int n) {
	float** m = (float**)aligned_alloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)aligned_alloc(32 * n * sizeof(float*), 32);
	}
	reset(m, n);
	return m;
}

float** proof_generate(int n) {
	ifstream inp("input.txt");
	inp >> n;
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

float** aligned_proof_generate(int n) {
	ifstream inp("input.txt");
	inp >> n;
	float** m = (float**)aligned_alloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)aligned_alloc(32 * n * sizeof(float*), 32);
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

void reset(float** a, int n)  // 检测程序正确性
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i][j] = rand() % 10;
			//out << a[i][j] << " ";
		}
		//out << endl;
	}
}

void c_reset(float** a, int n)  // 参考指导书中提出的保证结果正确的初始化方法
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++)
			a[i][j] = 0;
		a[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			a[i][j] = rand();
	}
	for (int k = 0; k < n; k++)
		for (int i = k + 1; i < n; i++)
			for (int j = 0; j < n; j++)
				a[i][j] += a[k][j];
}

void show(float** a, int n) {  // 用于观察程序运行结果
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << a[i][j] << " ";
			//out << a[i][j] << " ";
		}
		cout << endl;
		//out << endl;
	}

}

int main() {
	srand(time(0));
	//in >> n;
	cin >> n;
	out << n << endl;
	cout << "问题规模为" << n << "，算法的运行次数为" << lim << "，使用固定初始值" << endl;
	float time_use = 0;
	struct timeval start1, start2, start3, start4, start5, start6, start7, start8;
    struct timeval end1, end2, end3, end4, end5, end6, end7, end8;
	//-----------------------------------------------------------------

	//float** m1 = generate(n);
	float** m1 = proof_generate(n);
	//show(m1, n);

    gettimeofday(&start1,NULL); //gettimeofday(&start,&tz);结果一样
	GE(m1, n);
    gettimeofday(&end1,NULL); //gettimeofday(&start,&tz);结果一样
	cout << endl << endl << endl;
	//show(m1, n);
    time_use=(end1.tv_sec-start1.tv_sec)*1000000+(end1.tv_usec-start1.tv_usec);//微秒
	cout << "GE: " << time_use / 1000
		<< "ms" << endl;

	//-----------------------------------------------------------------

	//float** m2 = generate(n);
	float** m2 = proof_generate(n);
    gettimeofday(&start2,NULL); //gettimeofday(&start,&tz);结果一样
	C_GE(m2, n);
    gettimeofday(&end2,NULL); //gettimeofday(&start,&tz);结果一样
	time_use=(end2.tv_sec-start2.tv_sec)*1000000+(end2.tv_usec-start2.tv_usec);//微秒
	cout << "C_GE: " << time_use / 1000
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	//float** m3 = generate(n);
	//float** m3 = proof_generate(n);
	//float** m3 = aligned_generate(n);
	float** m3 = aligned_proof_generate(n);

    gettimeofday(&start3,NULL); //gettimeofday(&start,&tz);结果一样
	neon_GE(m3, n);
    gettimeofday(&end3,NULL); //gettimeofday(&start,&tz);结果一样
	time_use=(end3.tv_sec-start3.tv_sec)*1000000+(end3.tv_usec-start3.tv_usec);//微秒
	cout << "neon_GE: " << time_use / 1000
		<< "ms" << endl;
	//show(m3, n);

	//-----------------------------------------------------------------

	//float** m4 = generate(n);
	float** m4 = proof_generate(n);
	//float** m4 = aligned_generate(n);
	//float** m4 = aligned_proof_generate(n);
	//show(m4, n);
	//cout<<endl;
    gettimeofday(&start4,NULL); //gettimeofday(&start,&tz);结果一样
	neon_U_GE(m4, n);
    gettimeofday(&end4,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end4.tv_sec-start4.tv_sec)*1000000+(end4.tv_usec-start4.tv_usec);//微秒
	cout << "neon_U_GE: " << time_use / 1000
		<< "ms" << endl;
	//show(m4, n);

	//-----------------------------------------------------------------



	//float** m5 = generate(n);
	float** m5 = proof_generate(n);
	//float** m5 = aligned_generate(n);
	//float** m5 = aligned_proof_generate(n);

    gettimeofday(&start5,NULL); //gettimeofday(&start,&tz);结果一样
	neon_U_GE_Loop1(m5, n);
    gettimeofday(&end5,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end5.tv_sec-start5.tv_sec)*1000000+(end5.tv_usec-start5.tv_usec);//微秒
	cout << "neon_U_GE_Loop1: " << time_use / 1000
		<< "ms" << endl;
	//show(m5, n);

	//-----------------------------------------------------------------

	//float** m6 = generate(n);
	float** m6 = proof_generate(n);
	//float** m6 = aligned_generate(n);
	//float** m6 = aligned_proof_generate(n);

    gettimeofday(&start6,NULL); //gettimeofday(&start,&tz);结果一样
	neon_U_GE_Loop2(m6, n);
    gettimeofday(&end6,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end6.tv_sec-start6.tv_sec)*1000000+(end6.tv_usec-start6.tv_usec);//微秒
	cout << "neon_U_GE_Loop2: " << time_use / 1000
		<< "ms" << endl;
	//show(m6, n);

	//-----------------------------------------------------------------

	//float** m7 = generate(n);
	float** m7 = proof_generate(n);
	//float** m7 = aligned_generate(n);
	//float** m7 = aligned_proof_generate(n);

    gettimeofday(&start7,NULL); //gettimeofday(&start,&tz);结果一样
	neon_U_GE_Opt1(m7, n);
    gettimeofday(&end7,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end7.tv_sec-start7.tv_sec)*1000000+(end7.tv_usec-start7.tv_usec);//微秒
	cout << "neon_U_GE_Opt1: " << time_use / 1000
		<< "ms" << endl;
	//show(m7, n);

	//-----------------------------------------------------------------

	//float** m8 = generate(n);
	float** m8 = proof_generate(n);
	//float** m8 = aligned_generate(n);
	//float** m8 = aligned_proof_generate(n);

    gettimeofday(&start8,NULL); //gettimeofday(&start,&tz);结果一样
	neon_U_GE_fms(m8, n);
    gettimeofday(&end8,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end8.tv_sec-start8.tv_sec)*1000000+(end8.tv_usec-start8.tv_usec);//微秒
	cout << "neon_U_GE_fms: " << time_use / 1000
		<< "ms" << endl;
	//show(m7, n);

	//-----------------------------------------------------------------
}
