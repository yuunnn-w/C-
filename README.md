//手写数字识别程序源代码
//开发者：小阳   QQ：2214916637   有任何问题请联系开发者处理。
//本程序版权归开发者小阳所有，原则上仅限学习，任何单位及个人未经许可不得用于商业用途。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>
#include <time.h>
#include<conio.h>
#define SAMPLE_NUM 30//宏定义每个数字的样本个数，可以修改
double learningrate;//定义学习率
double result[10];//保存softmax输出结果的全局变量
struct parameter//网络参数结构体
{
    double kernel1[3][3];
    double kernel11[3][3];
    double kernel2[3][3];
    double kernel22[3][3];
    double kernel3[3][3];
    double kernel33[3][3];
    double firsthiddenlayer[1152][180];
    double secondhiddenlayer[180][45];
    double outhiddenlayer [45][10];
};
struct result//保存网络每一步输出结果的结构体，为反向传播计算梯度提供数据
{
    double picturedata[30][30];
    double firstcon[28][28];
    double firstcon1[28][28];
    double secondcon[26][26];
    double secondcon1[26][26];
    double thirdcon[24][24];
    double thirdcon1[24][24];
    double beforepool[1][1152];
    double firstmlp[1][180];
    double firstrelu[1][180];
    double secondmlp[1][45];
    double secondrelu[1][45];
    double outmlp[1][10];//全连接输出
    double result[10];//softmax输出
};
struct input//保存全部训练集的结构体，训练集为30*30像素图片
{
    double a[10][SAMPLE_NUM][30][30];
};
struct sample//保存每一个图片样本的结构体
{
    double a[30][30];
    int number;//样本标签，0~9之间的数字
}Sample[SAMPLE_NUM*10];
void printf_file2(struct parameter* parameter4)//训练中途打印局部最优参数的函数，使用参数时需要删掉汉字部分，改名为Network_parameter.bin
{
    FILE*fp;
    fp=fopen("Network_parameter中途局部最优.bin","wb");
    struct parameter* parameter1;
    parameter1=(struct parameter*)malloc(sizeof(struct parameter));
    (*parameter1)=(*parameter4);
    fwrite(parameter1,sizeof(struct parameter),1,fp);
    fclose(fp);
    free(parameter1);
    parameter1=NULL;
    return;
}
double Cross_entropy(double *a,int m)//计算每一次训练结果的交叉熵损失函数
{
    double u=0;
    u=(-log10(a[m]));
    return u;
}
void split(double *a,double *b,double *c)//把全连接反向传播来的梯度拆成两部分分别输入两个通道的函数
{
    int e=0;
    for(int i=0;i<576;i++)
    {
        a[i]=c[e];
        e++;
    }
    for(int i=0;i<576;i++)
    {
        b[i]=c[e];
        e++;
    }
}
void Matrix_bp(int m,int n,double *a,double *b)//更新网络参数的函数
{
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)a[i*n+j]-=learningrate*b[i*n+j];
}
void Matrix_multiplication(int m,int n,int p,double *a,double *b,double *c)//正向传播做矩阵乘法的函数
{
    for(int i=0;i<m;i++)
        for(int j=0;j<p;j++)
        {
            c[i*n+j]=0;
            for(int k=0;k<n;k++)c[i*n+j]+=a[i*n+k]*b[k*p+j];
        }
}
void Matrix_bp_mul(int m,int n,double *a,double *b,double *c)//反向传播时从后一层的梯度推导到前一层梯度的函数
{
    for(int i=0;i<m;i++)
    {
        c[i]=0;//空间清空，方便做累加
        for(int j=0;j<n;j++)
        {
            c[i]+=a[i*n+j]*b[j];
        }
    }
}
void Matrix_line_mul(int m,int n,double *a,double *b,double *c)//反向传播计算每一层网络参数矩阵梯度的函数（前一层神经元梯度行矩阵乘本层神经元梯度列矩阵，得到本层参数梯度）
{
    for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                c[i*n+j]=a[i]*b[j];
            }
        }
}
void Leakyrelu(int m,int n,double *a,double *b)//激活函数
{
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
        {
            (b[i*n+j])=max(a[i*n+j],a[i*n+j]*0.05);
        }
}
void bprelu(int m,double *a,double *b,double *c)//激活函数的反向传播
{
    for(int i=0;i<m;i++)
    {
        if(a[i]>0)c[i]=b[i]*1;
        else if(a[i]<=0)c[i]=b[i]*0.05;
    }
}
void Matrix_expansion(int m,int n,double *a,double *c,double *b)//把两个通道的卷积输出矩阵（24*24）展开并合并成一个1152长度的数组（向量）方便输入到全连接层
{
    int k=0;
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
        {
            b[k]=a[i*n+j];
            k++;
        }
     k=m*n;
     for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
        {
            b[k]=c[i*n+j];
            k++;
        }
}
void Convolution(int m,int n,int p,double *a,double *b,double *c)//做卷积运算的函数
{
    for(int i=0;i<(m-p+1);i++)
    {
        for(int j=0;j<(n-p+1);j++)
        {
            int u=0;
            c[i*(n-p+1)+j]=0;
            for(int k=i;k<i+3;k++)
            {
                int d=0;
                for(int l=j;l<j+3;l++)
                {
                    c[i*(n-p+1)+j]+=a[k*n+l]*b[u*p+d];
                    d++;
                }
                u++;
            }
        }
    }
}
void Overturn_convolution_kernel(int m,double *a,double *b)//卷积网络反向传播递推到前一层梯度时需要翻转卷积核然后和本层梯度padding后的矩阵做卷积，本函数是翻转卷积核的函数
{
    for(int i=0;i<m;i++)
    {
        int j=0;
        for(int k=m-1;k>=0;k--)
        {
            b[i*m+j]=a[i*m+k];
            j++;
        }
    }
        int j=0;
        for(int k=m-1;k>=0;k--)
        {
            for(int i=0;i<m;i++)
            {
                b[j*m+i]=a[k*m+i];
            }
            j++;
        }
}
int inputSample()//从图片中提取样本数据的函数
{
for(int m=0;m<10;m++)//数字0~9样本
{
 for(int i=0;i<SAMPLE_NUM;i++)//每个数字有SAMPLE_NUM个样本
 {
    char (*e);
    int (*l);
    e=(char*)malloc(sizeof(char)*120);
    l=(int*)malloc(sizeof(int)*960);
    char route_name1[5];
    char route_name2[30]="Training_set\\";
    sprintf(route_name1,"%d%s",m,"\\");
    strcat(route_name2,route_name1);
	FILE *fp;
	char file_name1[10];
	sprintf(file_name1,"%d%s",i+1,".bmp");//通过i++循环批量读取文件
	strcat(route_name2,file_name1);
	fp=fopen(route_name2,"rb");
	if(fp==NULL)
	{
		printf("训练集打开失败，请检查Training_set文件夹是否存在以及训练图片是否完整！\n");
		return 1;
	}
    fseek(fp,62,SEEK_SET);//bmp单色位图像素数据从62个字节处开始
    fread(e,sizeof(char),120,fp);//把所以数据以char型的格式读到e数组中
    fclose(fp);
    int y=0;
    for(int r=0;r<120;r++)
    {
      for (int u=1;u<9;u++)
      {
        l[y]=(int)(((e[r])) >> (8-u) & 0x01);//把每一个char型数据拆开成01数据存放到数组l中
        y++;
        if(y>960)break;
      };
    };
    int g=0;
    for(int u=0;u<30;u++)
    {
        y=0;
        for(int j=0;j<32;j++)
        {
            if((j!=30)&&(j!=31)){Sample[m*SAMPLE_NUM+i].a[u][y]=l[g];y++;};//去掉windows自动补0的数据，把真正的数据存放的样本结构体中
            g++;
        }
    }
    int q=Sample[m*SAMPLE_NUM+i].a[0][0];
    if(q==1)
/*由于182字节大小的30*30单色bmp位图采用白色为1，黑色为0的存储方式，而184字节大小的图片则恰好相反，所以这里要检测文件格式，
一般图片右下角不会有字迹，所以检测那里是0还是1即可得知格式，然后调整样本格式为1代表黑色，0白色的格式（图片大小为184字节），这样保证训练结果可靠性*/
    {
        int n=0;
        int z=0;
        for(int b=0;b<30;b++)
        {
            n=0;
            for(;;)
            {
                if(n>=30)break;
                if(Sample[m*SAMPLE_NUM+i].a[z][n]==0)Sample[m*SAMPLE_NUM+i].a[z][n]=1;
                else if(Sample[m*SAMPLE_NUM+i].a[z][n]==1)Sample[m*SAMPLE_NUM+i].a[z][n]=0;
                n++;
            }
            z++;
        }
    }
    Sample[m*SAMPLE_NUM+i].number=m;//给样本打标签
    free(e);
    e=NULL;
    free(l);
    l=NULL;
 }
}
    return 0;
}
void padding(int o,double *a,double *b)//反向传播时填充本层梯度的函数，相当于在梯度矩阵周围填充一圈0，这样和翻转后的卷积核做卷积后梯度矩阵大小才能恢复到前一层大小
{
    for(int i=0;i<(o+2);i++)
    {
        for(int j=0;j<(o+2);j++)b[i*(o+2)+j]=0;
    }
    int m=0;
    for(int k=2;k<o;k++)
    {
        b[k*(o+2)]=0;
        b[k*(o+2)+1]=0;
        int n=0;
        for(int j=2;j<o;j++)
        {
            b[k*(o+2)+j]=a[m*(o-2)+n];
            n++;
        }
        m++;
        b[k*(o+2)+o]=0;
        b[k*(o+2)+o+1]=0;
    }
}
void initialization(struct parameter *a)//用随机数初始化网络参数
{
    srand(time(NULL));
    for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)a->kernel1[j][k]=(rand()/(RAND_MAX+1.0));
    for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)a->kernel2[j][k]=(rand()/(RAND_MAX+1.0))/5;
    for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)a->kernel3[j][k]=(rand()/(RAND_MAX+1.0))/3;
    for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)a->kernel11[j][k]=(rand()/(RAND_MAX+1.0));
    for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)a->kernel22[j][k]=(rand()/(RAND_MAX+1.0))/5;
    for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)a->kernel33[j][k]=(rand()/(RAND_MAX+1.0))/3;
    for(int i=0;i<1152;i++)
        for(int j=0;j<180;j++)a->firsthiddenlayer[i][j]=(rand()/(RAND_MAX+1.0))/1000;
    for(int i=0;i<180;i++)
        for(int j=0;j<45;j++)a->secondhiddenlayer[i][j]=(rand()/(RAND_MAX+1.0))/100;
    for(int i=0;i<45;i++)
        for(int j=0;j<10;j++)a->outhiddenlayer[i][j]=(rand()/(RAND_MAX+1.0))/10;
}
void forward_propagating(struct result* data,double *a,struct parameter *c)//正向传播的函数
{
    for(int z=0;z<30;z++)
        for(int g=0;g<30;g++)data->picturedata[z][g]=a[z*30+g];//这个循环没用，只是为了调试网络时为了使输入图片可视化的操作，保存下图片数据
    Convolution(30,30,3,a,&c->kernel1[0][0],&data->firstcon[0][0]);//第一通道第一层卷积
    Convolution(30,30,3,a,&c->kernel11[0][0],&data->firstcon1[0][0]);//第二通道第一层卷积
    Convolution(28,28,3,&data->firstcon[0][0],&c->kernel2[0][0],&data->secondcon[0][0]);//第一通道第二层卷积
    Convolution(28,28,3,&data->firstcon1[0][0],&c->kernel22[0][0],&data->secondcon1[0][0]);//第二通道第二层卷积
    Convolution(26,26,3,&data->secondcon[0][0],&c->kernel3[0][0],&data->thirdcon[0][0]);//第一通道第三层卷积
    Convolution(26,26,3,&data->secondcon1[0][0],&c->kernel33[0][0],&data->thirdcon1[0][0]);//第二通道第三层卷积
    Matrix_expansion(24,24,&data->thirdcon[0][0],&data->thirdcon1[0][0],&data->beforepool[0][0]);//把卷积输出扩展成全连接输入
    Matrix_multiplication(1,1152,180,&data->beforepool[0][0],&c->firsthiddenlayer[0][0],&data->firstmlp[0][0]);//第一层全连接
    Leakyrelu(1,180,&data->firstmlp[0][0],&data->firstrelu[0][0]);//激活函数
    Matrix_multiplication(1,180,45,&data->firstrelu[0][0],&c->secondhiddenlayer[0][0],&data->secondmlp[0][0]);//第二层全连接
    Leakyrelu(1,45,&data->secondmlp[0][0],&data->secondrelu[0][0]);//激活函数
    Matrix_multiplication(1,45,10,&data->secondrelu[0][0],&c->outhiddenlayer[0][0],&data->outmlp[0][0]);//第三层全连接
    double p=0;
    for(int i=0;i<10;i++)//softmax分类器
    {
        p+=(exp(data->outmlp[0][i]));
    };
    for(int i=0; i<10; i++)
    {
         data->result[i]=exp(data->outmlp[0][i])/p;
         result[i]=data->result[i];//softmax输出
    };

    return;
}
void backPropagation(int a,struct result *data,struct parameter *parameter1)//反向传播batch梯度下降更新参数函数
/*
本函数内部过于复杂，我不做过多注释，其中bias结尾的变量代表每一层的梯度（用于计算传播到下一层的梯度），
wbias结尾的变量代表每一层网络参数的梯度（用于更新参数），relubias结尾的变量代表反向传播到激活层的梯度，
其中层数已经用first等序数词注明。本网络结构是两个通道的卷积加三层全连接，每个通道有三层卷积层，无池化层，
采用的是3*3卷积核，一共六个。全连接尺寸是1152，180，45，10，最后接上softmax输出十个概率判别值。
*/
{
    double *outbias;
    outbias=(double *)malloc(10*sizeof(double));
    for(int i=0;i<10;i++)//计算softmax交叉熵损失，然后计算传回来的梯度
    {
        if(i==a)outbias[i]=(data->result[i]-1);
        else outbias[i]=data->result[i];
    }
    double *outwbias;
    outwbias=(double *)malloc(450*sizeof(double));
    Matrix_line_mul(45,10,&data->secondrelu[0][0],outbias,outwbias);
    double *secondrelubias;
    secondrelubias=(double *)malloc(45*sizeof(double));
    Matrix_bp_mul(45,10,&parameter1->outhiddenlayer[0][0],outbias,secondrelubias);
    free(outbias);
    outbias=NULL;
    double *secondbias;
    secondbias=(double *)malloc(180*sizeof(double));
    bprelu(45,&data->secondmlp[0][0],secondrelubias,secondbias);
    free(secondrelubias);
    secondrelubias=NULL;
    double *secondwbias;
    secondwbias=(double *)malloc(8100*sizeof(double));
    Matrix_line_mul(180,45,&data->firstrelu[0][0],secondbias,secondwbias);
    double *firstrelubias;
    firstrelubias=(double *)malloc(180*sizeof(double));
    Matrix_bp_mul(180,45,&parameter1->secondhiddenlayer[0][0],secondbias,firstrelubias);
    free(secondbias);
    secondbias=NULL;
    double *firstbias;
    firstbias=(double *)malloc(180*sizeof(double));
    bprelu(180,&data->firstmlp[0][0],firstrelubias,firstbias);
    free(firstrelubias);
    firstrelubias=NULL;
    double *firstwbias;
    firstwbias=(double *)malloc(207360*sizeof(double));
    Matrix_line_mul(1152,180,&data->beforepool[0][0],firstbias,firstwbias);
    double *allconbias;
    allconbias=(double *)malloc(1152*sizeof(double));
    Matrix_bp_mul(1152,180,&parameter1->firsthiddenlayer[0][0],firstbias,allconbias);
    free(firstbias);
    firstbias=NULL;
    double *thirdconbias;
    thirdconbias=(double *)malloc(576*sizeof(double));
    double *thirdconbias1;
    thirdconbias1=(double *)malloc(576*sizeof(double));
    split(thirdconbias,thirdconbias1,allconbias);
    free(allconbias);
    allconbias=NULL;
    double *thirdkernelbias;
    thirdkernelbias=(double *)malloc(9*sizeof(double));
    Convolution(26,26,24,&data->secondcon[0][0],thirdconbias,thirdkernelbias);
    double *secondconbias;
    secondconbias=(double*)malloc(676*sizeof(double));
    double *turnthirdkerne3;
    turnthirdkerne3=(double *)malloc(9*sizeof(double));
    Overturn_convolution_kernel(3,&parameter1->kernel3[0][0],turnthirdkerne3);
    double *padthirdkernelbias;
    padthirdkernelbias=(double *)malloc(784*sizeof(double));
    padding(26,thirdconbias,padthirdkernelbias);
    free(thirdconbias);
    thirdconbias=NULL;
    Convolution(28,28,3,padthirdkernelbias,turnthirdkerne3,secondconbias);
    free(turnthirdkerne3);
    turnthirdkerne3=NULL;
    free(padthirdkernelbias);
    padthirdkernelbias=NULL;
     double *secondkernelbias;
    secondkernelbias=(double *)malloc(9*sizeof(double));
    Convolution(28,28,26,&data->firstcon[0][0],secondconbias,secondkernelbias);
    double *firstconbias;
    firstconbias=(double*)malloc(784*sizeof(double));
    double *turnsecondkernel;
    turnsecondkernel=(double *)malloc(9*sizeof(double));
    Overturn_convolution_kernel(3,&parameter1->kernel2[0][0],turnsecondkernel);
    double *padsecondkernelbias;
    padsecondkernelbias=(double *)malloc(900*sizeof(double));
    padding(28,secondconbias,padsecondkernelbias);
    free(secondconbias);
    secondconbias=NULL;
    Convolution(30,30,3,padsecondkernelbias,turnsecondkernel,firstconbias);
    free(turnsecondkernel);
    turnsecondkernel=NULL;
    free(padsecondkernelbias);
    padsecondkernelbias=NULL;
    double *firstkernelbias;
    firstkernelbias=(double *)malloc(9*sizeof(double));
    Convolution(30,30,28,&data->picturedata[0][0],firstconbias,firstkernelbias);
    free(firstconbias);
    firstconbias=NULL;
    double *thirdkernelbias1;
    thirdkernelbias1=(double *)malloc(9*sizeof(double));
    Convolution(26,26,24,&data->secondcon1[0][0],thirdconbias1,thirdkernelbias1);
    double *secondconbias1;
    secondconbias1=(double*)malloc(676*sizeof(double));
    double *turnthirdkernel1;
    turnthirdkernel1=(double *)malloc(9*sizeof(double));
    Overturn_convolution_kernel(3,&parameter1->kernel33[0][0],turnthirdkernel1);
    double *padthirdkernelbias1;
    padthirdkernelbias1=(double *)malloc(784*sizeof(double));
    padding(26,thirdconbias1,padthirdkernelbias1);
    free(thirdconbias1);
    thirdconbias1=NULL;
    Convolution(28,28,3,padthirdkernelbias1,turnthirdkernel1,secondconbias1);
    free(turnthirdkernel1);
    turnthirdkernel1=NULL;
    free(padthirdkernelbias1);
    padthirdkernelbias1=NULL;
    double *secondkernelbias1;
    secondkernelbias1=(double *)malloc(9*sizeof(double));
    Convolution(28,28,26,&data->firstcon1[0][0],secondconbias1,secondkernelbias1);
    double *firstconbias1;
    firstconbias1=(double*)malloc(784*sizeof(double));
    double *turnsecondkernel1;
    turnsecondkernel1=(double *)malloc(9*sizeof(double));
    Overturn_convolution_kernel(3,&parameter1->kernel22[0][0],turnsecondkernel1);
    double *padsecondkernelbias1;
    padsecondkernelbias1=(double *)malloc(900*sizeof(double));
    padding(28,secondconbias1,padsecondkernelbias1);
    free(secondconbias1);
    secondconbias=NULL;
    Convolution(30,30,3,padsecondkernelbias1,turnsecondkernel1,firstconbias1);
    free(turnsecondkernel1);
    turnsecondkernel1=NULL;
    free(padsecondkernelbias1);
    padsecondkernelbias1=NULL;
    double *firstkernelbias1;
    firstkernelbias1=(double *)malloc(9*sizeof(double));
    Convolution(30,30,28,&data->picturedata[0][0],firstconbias1,firstkernelbias1);
    free(firstconbias1);
    firstconbias1=NULL;
    Matrix_bp(3,3,&parameter1->kernel1[0][0],firstkernelbias);//iterations更新参数
    Matrix_bp(3,3,&parameter1->kernel2[0][0],secondkernelbias);
    Matrix_bp(3,3,&parameter1->kernel3[0][0],thirdkernelbias);
    Matrix_bp(3,3,&parameter1->kernel11[0][0],firstkernelbias1);
    Matrix_bp(3,3,&parameter1->kernel22[0][0],secondkernelbias1);
    Matrix_bp(3,3,&parameter1->kernel33[0][0],thirdkernelbias1);
    Matrix_bp(1152,180,&parameter1->firsthiddenlayer[0][0],firstwbias);
    Matrix_bp(180,45,&parameter1->secondhiddenlayer[0][0],secondwbias);
    Matrix_bp(45,10,&parameter1->outhiddenlayer[0][0],outwbias);
    free(firstkernelbias);//释放中途动态分配的变量，防止内存溢出
    firstkernelbias=NULL;
    free(secondkernelbias);
    secondkernelbias=NULL;
    free(thirdkernelbias);
    thirdkernelbias=NULL;
    free(firstkernelbias1);
    firstkernelbias1=NULL;
    free(secondkernelbias1);
    secondkernelbias1=NULL;
    free(thirdkernelbias1);
    thirdkernelbias1=NULL;
    free(firstwbias);
    firstwbias=NULL;
    free(secondwbias);
    secondwbias=NULL;
    free(outwbias);
    outwbias=NULL;
    return;
}
void learn(int m,struct result *data,struct parameter *parameter1)//学习函数
{
    double max2=2;//保存每一次训练的最大交叉熵
    for(int o=0;o<m;o++)//动态学习率
    {
    learningrate=pow((max2/10.0),1.7);
    if(learningrate>=0.01)learningrate=0.01;
    if((o+1)%10==0)//每十次训练（若有300个样本则是训练了3000次）刷新一次训练进度及学习率等数据
    {
     if(o!=9)printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
     if(100*((double)(o+1)/(double)m)<10)printf("训练进度: %lf",100*((double)(o+1)/(double)m));
     else printf("训练进度:%lf",100*((double)(o+1)/(double)m));
     if(max2<10)printf("%%  交叉熵损失: %lf  学习率:%.10lf",max2,learningrate);
     else if(max2>=10)printf("%%  交叉熵损失:%lf  学习率:%.10lf",max2,learningrate);
     if(learningrate<0.0000000001)printf_file2(parameter1);//如果找到局部最优则打印网络参数
    }
    int a,b;
    srand(time(NULL));//洗牌算法，用于打乱样本
    for(int q=0;q<300;q++)
    {
        a=(int)((rand()/(RAND_MAX+1.0))*300);//确定本轮随机交换的变量下标
        b=(int)((rand()/(RAND_MAX+1.0))*300);
        if(a>=0&&a<300&&(a!=b)&&b>=0&&b<300)
        {
            struct sample* sample5;
            sample5=(struct sample *)malloc(sizeof(struct sample));
            (*sample5)=Sample[a];
            Sample[a]=Sample[b];
            Sample[b]=(*sample5);
            free(sample5);
            sample5=NULL;
        }
        else continue;
    }
    for(int i=0;i<SAMPLE_NUM*10;i++)//训练已经打乱的所有样本
    {
            max2=0;
            struct sample* sample3;
            sample3=(struct sample *)malloc(sizeof(struct sample));
            (*sample3)=Sample[i];
            int y=sample3->number;
            forward_propagating(data,&sample3->a[0][0],parameter1);//正向传播
            backPropagation(y,data,parameter1);//反向传播
            free(sample3);
            sample3=NULL;
            double g=Cross_entropy(&data->result[0],y);//计算本轮最大交叉熵损失，用于指导调整学习率
            if(g>max2)max2=g;
    }
    }
    printf("\n");
    return;
}
void test_network(struct result *data2,struct parameter* parameter2)//用测试集中的样本测试网络，一共有十个测试样本
{
char e[120];
int l[960];
double data[30][30];
for(int i=0;i<10;i++)
{
	FILE *fp;
	char s[30];
	sprintf(s,"%s%d%s","Training_set//Test_set//",i+1,".bmp");
	printf("\n打开的文件名:%s\n",s);
	fp = fopen(s, "rb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		system("pause");
		return;
	}
    fseek(fp, 62, SEEK_SET);
    fread(e,sizeof(char),120,fp);
    fclose(fp);
    int y=0;
    for(int r=0;r<120;r++)
    {
      for (int u=1;u<9;u++)
      {
        l[y]=(int)((e[r]) >> (8-u) & 0x01);
        y++;
        if(y>960)break;
      };
    };
    y=0;
    int g=0;
    for(int u=0;u<30;u++)
    {
        y=0;
        for(int j=0;j<32;j++)
        {
            if((j!=30)&&(j!=31)){data[u][y]=l[g];y++;};
            g++;
        }
    }
    int q=data[0][0];
    if(q==1)
    {
        int n=0;
        int z=0;
        for(int b=0;b<30;b++)
        {
            n=0;
            for(;;)
            {
                if(n>=30)break;
                if(data[z][n]==0)data[z][n]=1;
                else if(data[z][n]==1)data[z][n]=0;
                n++;
            }
            z++;
        }
    }
    forward_propagating(data2,&data[0][0],parameter2);//把获取的样本数据正向传播一次
    double sum=0;
    int k=0;
    for(int j=0;j<10;j++)
        {
            if(result[j]>sum)
            {
                sum=result[j];
                k=j;//获取分类结果
            }
            else continue;
        }
    printf("\n");
    for(int i=0;i<10;i++)//打印分类结果
    {
        printf("预测值是%d的概率：%lf\n",i,result[i]);
    }
    printf("最终预测值:%d\n",k);
}
return ;
}
int read_file(struct parameter* parameter4)//用于训练前读取网络参数的函数
{
    FILE*fp;
    fp=fopen("Training_set//Network_parameter.bin","rb");
    if(fp==NULL)
    {
        printf("文件打开失败，请检查网络参数文件是否在训练集文件夹内！\n");
        return 1;
    }
    struct parameter* parameter1;
    parameter1=(struct parameter*)malloc(sizeof(struct parameter));
    fread(parameter1,sizeof(struct parameter),1,fp);
    (*parameter4)=(*parameter1);
    fclose(fp);
    free(parameter1);
    parameter1=NULL;
    return 0;
}
void printf_file(struct parameter* parameter4)//用于训练结束后保存网络参数的函数
{
    FILE*fp;
    fp=fopen("Training_set//Network_parameter.bin","wb");//采用二进制格式保存参数，便于读取
    struct parameter* parameter1;
    parameter1=(struct parameter*)malloc(sizeof(struct parameter));
    (*parameter1)=(*parameter4);
    fwrite(parameter1,sizeof(struct parameter),1,fp);//打印网络结构体
    fclose(fp);
    free(parameter1);
    parameter1=NULL;
    return;
}
int main()//主函数
{
    printf("欢迎使用手写数字卷积神经网络训练程序\n");
    printf("开发者：小阳   QQ：2214916637   有任何问题请联系开发者处理。\n");
    printf("本软件版权归开发者小阳所有，原则上仅限学习，任何单位及个人未经许可不得用于商业用途。\n\n");
    printf("请先将Training_set训练集文件夹放在程序目录下。\n");
    int h=inputSample();//提取样本数据
    if(h==0)printf("训练数据读取成功\n");
    else if(h==1)
    {
        printf("训练集读取失败！程序自动退出\n");
        system("pause");
        return 0;
    }
    struct parameter *storage;//定义存放网络参数的结构体
    (storage) = (struct parameter*)malloc(sizeof(struct parameter));//动态分配空间
    struct result *data;
    (data) = (struct result*)malloc(sizeof(struct result));
    iiii:
    printf("请问您是否希望从已训练的网络参数文件中读取网络参数？是请按y，否请按n。\n");
    setbuf(stdin,NULL);//清空键盘缓冲区
    char g;
    g=getch();
    if(g=='y')
    {
        int h=read_file(storage);
        if(h==1)
        {
                printf("参数包不存在！开始自动随机初始化网络参数\n");
                initialization(storage);
                printf_file(storage);
                printf("网络参数初始化完毕！\n");
        }
        if(h==0)printf("参数读取成功！\n");
    }
    else if(g=='n')
    {
        initialization(storage);
        printf_file(storage);
        printf("网络参数初始化完毕！\n");
    }
    else goto iiii;
    oooo:
    printf("请输入您想训练的次数:\n");
    int d;
    scanf("%d",&d);
    printf("开始训练\n");
    learn(d,data,storage);//开始训练
    test_network(data,storage);//测试网络
    printf_file(storage);
    kkkk:
    printf("继续训练请按1，退出请按2\n");
    setbuf(stdin,NULL);
    char v;
    v=getch();
    if(v=='1')goto oooo;
    else if(v=='2'){printf_file(storage);return 0;}//退出则在退出之前保存网络参数
    else goto kkkk;
    return 0;
}
