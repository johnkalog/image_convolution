#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stddef.h>

#ifndef RGB_PROCESS_H_
#define RGB_PROCESS_H_

typedef struct RGB_Pixel  //3 byte
{
  unsigned char r;
  unsigned char g;
  unsigned char b;
}RGB_Pixel;

typedef struct RGB_Process
{
  unsigned int x_image_size;
  unsigned int y_image_size;

  RGB_Pixel** Image_Array;
  RGB_Pixel** New_Image_Array;
  RGB_Pixel** tmp;

  unsigned char* oneD_Array;

  int rank;
  int x_pos;
  int y_pos;

  int neighbours_num;

  int neighbour_proc_ranks[8];
  MPI_Request *Send_r;
  MPI_Request *Receiv_r;

  RGB_Pixel* Received;
  RGB_Pixel* SentArrays[8];

  MPI_File mf,nf;
  MPI_Datatype filetype;

}RGB_Process;


static inline void Change_Image_RGB(RGB_Process* P){
  int index=0,i,j;


  for(i=1;i<=P->x_image_size;i++)
  {
    for(j=1;j<=P->y_image_size;j++)
    {
      P->oneD_Array[index]=P->Image_Array[i][j].r;
      P->oneD_Array[index+1]=P->Image_Array[i][j].g;
      P->oneD_Array[index+2]=P->Image_Array[i][j].b;
      index=index+3;
    }
  }

  MPI_File_open(MPI_COMM_WORLD,"new_rgb.raw",MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&P->nf);

  MPI_File_set_view(P->nf, 0, MPI_UNSIGNED_CHAR, P->filetype, "native",MPI_INFO_NULL);
  MPI_File_write_all(P->nf, P->oneD_Array, P->x_image_size*P->y_image_size*3, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);

  MPI_File_close(&P->nf);

}


void DestroyRGB_Process(RGB_Process* P)
{
  for(int i=0;i<=P->x_image_size+1;i++)
  {
      free(P->Image_Array[i]);
      free(P->tmp[i]);
  }
  free(P->oneD_Array);
  free(P->Image_Array);
  free(P->tmp);

  free(P->Send_r);
  free(P->Receiv_r);
  free(P->Received);
}





static inline void Read_ImageRGB(char* filename,RGB_Process* P,int x,int y,int allp)
{

  int index,i,j;
  int gsizes[2],distribs[2],dargs[2],psizes[2];


  P->oneD_Array=malloc(P->x_image_size*P->y_image_size*sizeof(RGB_Pixel));
/////////MPI SYS/////////////////////////////////////////
  MPI_File_open(MPI_COMM_WORLD,filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&P->mf);

  gsizes[0] = x;
  gsizes[1] = y*3;

  distribs[0] = MPI_DISTRIBUTE_BLOCK; /* block distribution */
  distribs[1] = MPI_DISTRIBUTE_BLOCK; /* block distribution */
  dargs[0] = MPI_DISTRIBUTE_DFLT_DARG; /* default block size */
  dargs[1] = MPI_DISTRIBUTE_DFLT_DARG; /* default block size */

  psizes[0] = y/P->y_image_size;
  psizes[1] = x/P->x_image_size;


  MPI_Type_create_darray(allp, P->rank, 2, gsizes, distribs, dargs,
  psizes, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &P->filetype);
  MPI_Type_commit(&P->filetype);
  MPI_File_set_view(P->mf, 0, MPI_UNSIGNED_CHAR, P->filetype, "native",
  MPI_INFO_NULL);
  MPI_File_read_all(P->mf, P->oneD_Array, P->x_image_size*P->y_image_size*3, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
  MPI_File_close(&P->mf);
/////////////////////////////////////////////////////////////////////////////////

  P->Image_Array=malloc((P->x_image_size+2)*sizeof(RGB_Pixel*));
  index=0;
  for(i=0;i<=P->x_image_size+1;i++)
  {
    P->Image_Array[i]=malloc((P->y_image_size+2)*sizeof(RGB_Pixel));
    for(j=1;j<=P->y_image_size;j++)
    {
      if(i!=0 && i!=(P->x_image_size+1))
      {
        P->Image_Array[i][j].r=P->oneD_Array[index];
        P->Image_Array[i][j].g=P->oneD_Array[index+1];
        P->Image_Array[i][j].b=P->oneD_Array[index+2];
        index=index+3;
      }
    }
  }
}


void insideFilterApply_RGB(RGB_Process* P,double h[3][3]){
  unsigned char sum;
  int x_index,y_index;
  for (int i=2; i<=P->x_image_size-1; i++ ){
    for (int j=2; j<=P->y_image_size-1; j++ ){
      sum = 0;
      x_index=0;
      for (int k=-1; k<2; k++ ){
        y_index = 0;
        for (int l=-1; l<2; l++ ){
          sum =sum + P->Image_Array[i+k][j+l].r*h[x_index][y_index];
          y_index++;
        }
        x_index++;
      }
      P->New_Image_Array[i][j].r = sum;

      sum = 0;
      x_index=0;
      for (int k=-1; k<2; k++ ){
        y_index = 0;
        for (int l=-1; l<2; l++ ){
          sum =sum + P->Image_Array[i+k][j+l].g*h[x_index][y_index];
          y_index++;
        }
        x_index++;
      }
      P->New_Image_Array[i][j].g = sum;

      sum = 0;
      x_index=0;
      for (int k=-1; k<2; k++ ){
        y_index = 0;
        for (int l=-1; l<2; l++ ){
          sum =sum + P->Image_Array[i+k][j+l].b*h[x_index][y_index];
          y_index++;
        }
        x_index++;
      }
      P->New_Image_Array[i][j].b = sum;
    }
  }
}


void outsideFilterApply_RGB(RGB_Process* P,double h[3][3]){
  int i,j,k,l;
  RGB_Pixel sum;
  int flag=0;

  sum.r=0;
  sum.g=0;
  sum.b=0;

  for(i=1;i<=P->x_image_size;i++)
  {
    if(i==1 || i==P->x_image_size)
    {
      for(j=1;j<=P->y_image_size;j++)
      {
        for(k=0;k<=2;k++)
        {
          for(l=0;l<=2;l++)
          {
            flag=0;
            if(i==1)
            {
              if(j==1)
              {
                if(k==0 && l==0)
                {
                  if(P->neighbour_proc_ranks[0]==-1)
                    {flag=1;}
                }
                else if(k==0 && l==1)
                {
                  if(P->neighbour_proc_ranks[1]==-1)
                  {flag=1;}
                }
                else if(k==1 && l==0)
                {
                  if(P->neighbour_proc_ranks[3]==-1)
                  {flag=1;}
                }
                else if(k==2 && l==0)
                {
                  if(P->neighbour_proc_ranks[3]==-1)
                  {flag=1;}
                }
                else if(k==0 && l==2)
                {
                  if(P->neighbour_proc_ranks[1]==-1)
                  {flag=1;}
                }
              }
              else if(j>1 && j<P->y_image_size)
              {
                if(P->neighbour_proc_ranks[1]==-1 && k==0)
                {flag=1;}
              }
              else if(j==P->y_image_size)
              {
                if(k==0 && l==1)
                {
                  if(P->neighbour_proc_ranks[1]==-1)
                  {flag=1;}
                }
                else if(k==0 && l==2)
                {
                  if(P->neighbour_proc_ranks[2]==-1)
                  {flag=1;}
                }
                else if(k==1 && l==2)
                {
                  if(P->neighbour_proc_ranks[4]==-1)
                  {flag=1;}
                }
                else if(k==2 && l==2)
                {
                  if(P->neighbour_proc_ranks[4]==-1)
                  {flag=1;}
                }
                else if(k==0 && l==0)
                {
                  if(P->neighbour_proc_ranks[1]==-1)
                  {flag=1;}
                }
              }
            }
            else if(i==P->x_image_size)
            {
              if(j==1)
              {
                if(k==1 && l==0)
                {
                  if(P->neighbour_proc_ranks[3]==-1)
                    {flag=1;}
                }
                else if(k==2 && l==0)
                {
                  if(P->neighbour_proc_ranks[5]==-1)
                  {flag=1;}
                }
                else if(k==2 && l==1)
                {
                  if(P->neighbour_proc_ranks[6]==-1)
                  {flag=1;}
                }
                else if(k==2 && l==2)
                {
                  if(P->neighbour_proc_ranks[6]==-1)
                  {flag=1;}
                }
                else if(k==0 && l==0)
                {
                  if(P->neighbour_proc_ranks[3]==-1)
                  {flag=1;}
                }
              }
              else if(j>1 && j<P->y_image_size)
              {
                if(P->neighbour_proc_ranks[6]==-1 && k==2)
                {flag=1;}
              }
              else if(j==P->y_image_size)
              {
                if(k==1 && l==2)
                {
                  if(P->neighbour_proc_ranks[4]==-1)
                    {flag=1;}
                }
                else if(k==2 && l==2)
                {
                  if(P->neighbour_proc_ranks[7]==-1)
                  {flag=1;}
                }
                else if(k==2 && l==1)
                {
                  if(P->neighbour_proc_ranks[6]==-1)
                  {flag=1;}
                }
                else if(k==0 && l==2)
                {
                  if(P->neighbour_proc_ranks[4]==-1)
                  {flag=1;}
                }
                else if(k==2 && l==0)
                {
                  if(P->neighbour_proc_ranks[6]==-1)
                  {flag=1;}
                }
              }
            }
            if(flag)
            {
                sum.r=sum.r+P->Image_Array[i][j].r*h[k][l];
                sum.g=sum.g+P->Image_Array[i][j].g*h[k][l];
                sum.b=sum.b+P->Image_Array[i][j].b*h[k][l];
            }
            else
            {
                sum.r=sum.r+P->Image_Array[i+k-1][j+l-1].r*h[k][l];
                sum.g=sum.g+P->Image_Array[i+k-1][j+l-1].g*h[k][l];
                sum.b=sum.b+P->Image_Array[i+k-1][j+l-1].b*h[k][l];
            }
          }
        }
        P->New_Image_Array[i][j].r=sum.r;
        P->New_Image_Array[i][j].g=sum.g;
        P->New_Image_Array[i][j].b=sum.b;
        sum.r=0;
        sum.g=0;
        sum.b=0;
      }
    }
    else
    {
      for(j=1;j<=P->y_image_size;j=j+P->y_image_size-1)
      {
        for(k=0;k<=2;k++)
        {
          for(l=0;l<=2;l++)
          {
            flag=0;
            if((P->neighbour_proc_ranks[3]==-1 && j==1 && l==0) || (P->neighbour_proc_ranks[4]==-1 && j==P->y_image_size && l==2))
            {flag=1;}
            if(flag)
            {
                sum.r=sum.r+P->Image_Array[i][j].r*h[k][l];
                sum.g=sum.g+P->Image_Array[i][j].g*h[k][l];
                sum.b=sum.b+P->Image_Array[i][j].b*h[k][l];
            }
            else
            {
                sum.r=sum.r+P->Image_Array[i+k-1][j+l-1].r*h[k][l];
                sum.g=sum.g+P->Image_Array[i+k-1][j+l-1].g*h[k][l];
                sum.b=sum.b+P->Image_Array[i+k-1][j+l-1].b*h[k][l];
            }
          }
        }
        P->New_Image_Array[i][j].r=sum.r;
        P->New_Image_Array[i][j].g=sum.g;
        P->New_Image_Array[i][j].b=sum.b;
        sum.r=0;
        sum.g=0;
        sum.b=0;
      }
    }
  }
}

void InitializeProcess_RGB(RGB_Process* P,char* filename ,int rank,int x,int y,int proc_num)
{
  int index=0;
  int divis,i,j,curr_i,curr_j;

  divis=(int)sqrt((double)proc_num);

  P->x_image_size=x/divis;
  P->y_image_size=y/divis;
  P->x_pos=rank/divis;
  P->y_pos=rank%divis;
    P->rank=rank;
  Read_ImageRGB(filename,P,x,y,proc_num);
  P->tmp=P->Image_Array;


  P->New_Image_Array=(RGB_Pixel**)malloc((P->x_image_size+2)*sizeof(RGB_Pixel *)); //giati se epomenh allagh tha parei thn thesi tou paliou

  	for(unsigned int i=0;i<=P->x_image_size+1;i++)
  	{
  		P->New_Image_Array[i]=(RGB_Pixel*)malloc((P->y_image_size+2)*sizeof(RGB_Pixel));
    }
  P->neighbours_num=0;


  for(i=-1;i<=1;i++)
  {
    for(j=-1;j<=1;j++)
    {
      if(i!=0 || j!=0)
      {
        curr_i=P->x_pos+i;
        curr_j=P->y_pos+j;
        if(!(curr_i <0 || curr_j<0 || curr_i >(divis -1) || curr_j>(divis -1)))
        {
          P->neighbour_proc_ranks[index]=divis*curr_i+curr_j;
          P->neighbours_num++;
        }
        else
        {
          P->neighbour_proc_ranks[index]=-1;
        }
        P->SentArrays[index]=NULL;
        index++;
      }
    }
  }
  int total=4+2*P->x_image_size+2*P->y_image_size;

  P->Received=malloc(total*sizeof(RGB_Pixel));
  P->Receiv_r=malloc(P->neighbours_num*sizeof(MPI_Request));
  P->Send_r=malloc(P->neighbours_num*sizeof(MPI_Request));
}

void SetImagePart_RGB(RGB_Process* P,int x_index,int y_index,RGB_Pixel* Image_part)
{

  int i;

  if(x_index==-1 && y_index==-1)                //NW
  {

    P->Image_Array[0][0]=Image_part[0];
  }
  else if(x_index==-1 && y_index==0)          //N
  {

    for(i=1;i<=P->y_image_size;i++)
    {
      P->Image_Array[0][i]=Image_part[i-1];
    }
  }
  else if(x_index==-1 && y_index==1)        //NE
  {
    P->Image_Array[0][P->y_image_size+1]=Image_part[0];
  }
  else if(x_index==0 && y_index==-1)      //W
  {

    for(i=1;i<=P->x_image_size;i++)
    {
      P->Image_Array[i][0]=Image_part[i-1];
    }
  }
  else if(x_index==0 && y_index==1)   //E
  {
    for(i=1;i<=P->x_image_size;i++)
    {
      P->Image_Array[i][P->y_image_size+1]=Image_part[i-1];
    }
  }
  else if(x_index==1 && y_index==-1)  //SW
  {
    P->Image_Array[P->x_image_size+1][0]=Image_part[0];
  }
  else if(x_index==1 && y_index==0) //S
  {

    for(i=1;i<=P->y_image_size;i++)
    {
      P->Image_Array[P->x_image_size+1][i]=Image_part[i-1];
    }
  }
  else if(x_index==1 && y_index==1)     //SE
  {
    P->Image_Array[P->x_image_size+1][P->y_image_size+1]=Image_part[0];
  }
}


void SetParts_RGB(RGB_Process *P)
{
  int i,j,index=0,size,array_index=0;


  for(i=-1;i<=1;i++)
  {
    for(j=-1;j<=1;j++)
    {
      if(i!=0 || j!=0)
      {
        if((abs(i)+abs(j))==2)
        {
          size=1;
        }
        else if(j==0)
        {
          size=P->y_image_size;
        }
        else if(i==0)
        {
          size=P->x_image_size;
        }
        if(P->neighbour_proc_ranks[index]!=-1)
        {
          SetImagePart_RGB(P,i,j,&P->Received[array_index]);
        }
        array_index+=size;
        index++;
      }
    }
  }
}



RGB_Pixel* GetImagePart_RGB(int x_index,int y_index,RGB_Process* P,int *size)
{
  RGB_Pixel* Part;


  if(x_index==-1 && y_index==-1)                //NW
  {
    *size=1*sizeof(RGB_Pixel);
    Part=malloc(*size);
    Part[0]=P->Image_Array[1][1];
    P->SentArrays[0]=Part;
  }
  else if(x_index==-1 && y_index==0)          //N
  {
    *size=P->y_image_size*sizeof(RGB_Pixel);
    Part=malloc(*size);
    for(int i=1;i<=P->y_image_size;i++)
    {
      Part[i-1]=P->Image_Array[1][i];
    }
      P->SentArrays[1]=Part;
  }
  else if(x_index==-1 && y_index==1)        //NE
  {
    *size=1*sizeof(RGB_Pixel);
    Part=malloc(*size);
    Part[0]=P->Image_Array[1][P->y_image_size];
    P->SentArrays[2]=Part;
  }
  else if(x_index==0 && y_index==-1)      //W
  {
    *size=P->x_image_size*sizeof(RGB_Pixel);
    Part=malloc(*size);
    for(int i=1;i<=P->x_image_size;i++)
    {
      Part[i-1]=P->Image_Array[i][1];
    }
      P->SentArrays[3]=Part;
  }
  else if(x_index==0 && y_index==1)   //E
  {
    *size=P->x_image_size*sizeof(RGB_Pixel);
    Part=malloc(*size);
    for(int i=1;i<=P->x_image_size;i++)
    {
      Part[i-1]=P->Image_Array[i][P->y_image_size];
    }
      P->SentArrays[4]=Part;
  }
  else if(x_index==1 && y_index==-1)  //SW
  {
    *size=1*sizeof(RGB_Pixel);
    Part=malloc(*size);
    Part[0]=P->Image_Array[P->x_image_size][1];
      P->SentArrays[5]=Part;
  }
  else if(x_index==1 && y_index==0) //S
  {
    *size=P->y_image_size*sizeof(RGB_Pixel);
    Part=malloc(*size);
    for(int i=1;i<=P->y_image_size;i++)
    {
      Part[i-1]=P->Image_Array[P->x_image_size][i];
    }
      P->SentArrays[6]=Part;
  }
  else if(x_index==1 && y_index==1)     //SE
  {
    *size=1*sizeof(RGB_Pixel);
    Part=malloc(*size);
    Part[0]=P->Image_Array[P->x_image_size][P->y_image_size];
      P->SentArrays[7]=Part;
  }
  return Part;
}


void Send_Arrays_RGB(RGB_Process *P)
{

  int i,j,index=0,neig_index=0,size;
  RGB_Pixel* part;


  for(i=-1;i<=1;i++)
  {
    for(j=-1;j<=1;j++)
    {
      if(i!=0 || j!=0)
      {
        if(P->neighbour_proc_ranks[index]!=-1)
        {

          part=GetImagePart_RGB(i,j,P,&size);

          MPI_Isend(part,size,MPI_UNSIGNED_CHAR,P->neighbour_proc_ranks[index],P->rank,MPI_COMM_WORLD,&P->Send_r[neig_index]);
          neig_index++;
        }
        index++;
      }
    }
  }

}



void Receive_Arrays_RGB(RGB_Process* P)
{
  int i,j,index=0,neig_index=0,size,array_index=0;

  for(i=-1;i<=1;i++)
  {
    for(j=-1;j<=1;j++)
    {
      if(i!=0 || j!=0)
      {
        if((abs(i)+abs(j))==2)
        {
          size=1;
        }
        else if(j==0)
        {
          size=P->y_image_size;
        }
        else if(i==0)
        {
          size=P->x_image_size;
        }
        if(P->neighbour_proc_ranks[index]!=-1)
        {

          MPI_Irecv(&P->Received[array_index],size*sizeof(RGB_Pixel),MPI_UNSIGNED_CHAR,P->neighbour_proc_ranks[index],P->neighbour_proc_ranks[index],MPI_COMM_WORLD,&P->Receiv_r[neig_index]);

          neig_index++;
        }
        array_index+=size;
        index++;
        }
      }
  }

}

#endif
