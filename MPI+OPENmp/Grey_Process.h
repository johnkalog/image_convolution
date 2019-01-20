#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>

#ifndef GREY_PROCESS_H_
#define GREY_PROCESS_H_

typedef unsigned char Pixel;



typedef struct Grey_Process
{
  unsigned int x_image_size;
  unsigned int y_image_size;

  Pixel** Image_Array;
  Pixel** New_Image_Array;
  Pixel** tmp;

  Pixel* oneD_Array;

  int rank;
  int x_pos;
  int y_pos;

  int neighbours_num;

  int neighbour_proc_ranks[8];
  MPI_Request *Send_r;
  MPI_Request *Receiv_r;

  Pixel* Received;
  Pixel* SentArrays[8];


  MPI_File mf,nf;
  MPI_Datatype filetype;


}Grey_Process;
static inline void Change_Image(Grey_Process* P){
  int index=0,i,j;


  for(i=1;i<=P->x_image_size;i++)
  {
    for(j=1;j<=P->y_image_size;j++)
    {
      P->oneD_Array[index]=P->Image_Array[i][j];
      index++;
    }
  }

  MPI_File_open(MPI_COMM_WORLD,"new_grey.raw",MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&P->nf);

  MPI_File_set_view(P->nf, 0, MPI_UNSIGNED_CHAR, P->filetype, "native",MPI_INFO_NULL);
  MPI_File_write_all(P->nf, P->oneD_Array, P->x_image_size*P->y_image_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);

  MPI_File_close(&P->nf);

}

static inline void DestroyGrey_Process(Grey_Process* P)
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

static inline void Read_Image(char* filename,Grey_Process* P,int x,int y,int allp)
{

  int index,i,j;
  int gsizes[2],distribs[2],dargs[2],psizes[2];


  P->oneD_Array=malloc(P->x_image_size*P->y_image_size*sizeof(unsigned char));
/////////MPI SYS/////////////////////////////////////////
  MPI_File_open(MPI_COMM_WORLD,filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&P->mf);

  gsizes[0] = x;
  gsizes[1] = y;

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
  MPI_File_read_all(P->mf, P->oneD_Array, P->x_image_size*P->y_image_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
/////////////////////////////////////////////////////////////////////////////////
  MPI_File_close(&P->mf);

  P->Image_Array=malloc((P->x_image_size+2)*sizeof(unsigned char*));
  index=0;

  for(i=0;i<=P->x_image_size+1;i++)
  {
    P->Image_Array[i]=malloc((P->y_image_size+2)*sizeof(unsigned char));
    for(j=1;j<=P->y_image_size;j++)
    {
      if(i!=0 && i!=(P->x_image_size+1))
      {
        P->Image_Array[i][j]=P->oneD_Array[index];
        index++;
      }

    }
  }
}

static inline void insideFilterApply(Grey_Process* P,double h[3][3]){
  unsigned char sum;
  int x_index,y_index;

#pragma omp parallel for collapse(2) private(x_index,y_index,sum) num_threads(num_of_threads)
  for (int i=2; i<=P->x_image_size-1; i++ ){
    for (int j=2; j<=P->y_image_size-1; j++ ){
      sum = 0;
      x_index=0;
      for (int k=-1; k<2; k++ ){
        y_index = 0;
        for (int l=-1; l<2; l++ ){
          sum =sum + P->Image_Array[i+k][j+l]*h[x_index][y_index];
          y_index++;
        }
        x_index++;
      }

      P->New_Image_Array[i][j] = sum;
    }
  }
}

static inline void outsideFilterApply(Grey_Process* P,double h[3][3])
{
  int i,j,k,l;
  unsigned char sum=0;
  int flag=0;

#pragma omp parallel for private(i,j,k,l,sum,flag) num_threads(num_of_threads)
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
                sum=sum+P->Image_Array[i][j]*h[k][l];
            }
            else
            {
                sum=sum+P->Image_Array[i+k-1][j+l-1]*h[k][l];
            }
          }
        }
        P->New_Image_Array[i][j]=sum;
        sum=0;
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
                sum=sum+P->Image_Array[i][j]*h[k][l];
            }
            else
            {
                sum=sum+P->Image_Array[i+k-1][j+l-1]*h[k][l];
            }
          }
        }
        P->New_Image_Array[i][j]=sum;
        sum=0;
      }
    }
  }
}



static inline void InitializeGrey_Process(Grey_Process* P,char* filename ,int rank,int x,int y,int proc_num)
{
  int index=0;
  int divis,i,j,curr_i,curr_j;

  divis=(int)sqrt((double)proc_num);

  P->x_image_size=x/divis;
  P->y_image_size=y/divis;
  P->x_pos=rank/divis;
  P->y_pos=rank%divis;
  P->rank=rank;

  Read_Image(filename,P,x,y,proc_num);
  P->tmp=P->Image_Array;


  P->New_Image_Array=(Pixel**)malloc((P->x_image_size+2)*sizeof(Pixel*)); //giati se epomenh allagh tha parei thn thesi tou paliou
#pragma omp parallel for private(i,j) num_threads(num_of_threads)
  for(i=0;i<=P->x_image_size+1;i++)
  {
    P->New_Image_Array[i]=malloc((P->y_image_size+2)*sizeof(Pixel));
    for(j=0;j<=P->y_image_size+1;j++)
    {
        P->New_Image_Array[i][j]='0';
    }
  }
  P->neighbours_num=0;



  index=0;
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


  P->Received=malloc(total*sizeof(Pixel));


  P->Receiv_r=malloc(P->neighbours_num*sizeof(MPI_Request));
  P->Send_r=malloc(P->neighbours_num*sizeof(MPI_Request));

}

static inline void SetImagePart(Grey_Process* P,int x_index,int y_index,Pixel* Image_part)
{

  if(x_index==-1 && y_index==-1)                //NW
  {

    P->Image_Array[0][0]=Image_part[0];
  }
  else if(x_index==-1 && y_index==0)          //N
  {

    for(int i=1;i<=P->y_image_size;i++)
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

    for(int i=1;i<=P->x_image_size;i++)
    {
      P->Image_Array[i][0]=Image_part[i-1];
    }
  }
  else if(x_index==0 && y_index==1)   //E
  {

    for(int i=1;i<=P->x_image_size;i++)
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

    for(int i=1;i<=P->y_image_size;i++)
    {
      P->Image_Array[P->x_image_size+1][i]=Image_part[i-1];
    }
  }
  else if(x_index==1 && y_index==1)     //SE
  {
    P->Image_Array[P->x_image_size+1][P->y_image_size+1]=Image_part[0];
  }
}

static inline void SetParts(Grey_Process *P)
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
          SetImagePart(P,i,j,&P->Received[array_index]);
        }
        array_index+=size;
        index++;
      }
    }
  }
}

static inline Pixel* GetImagePart(int x_index,int y_index,Grey_Process* P,int *size)
{
  Pixel* Part;

  if(x_index==-1 && y_index==-1)                //NW
  {
    *size=1;
    Part=malloc(1*sizeof(Pixel));
    Part[0]=P->Image_Array[1][1];
    P->SentArrays[0]=Part;
  }
  else if(x_index==-1 && y_index==0)          //N
  {
    *size=P->y_image_size;
    Part=malloc(P->y_image_size*sizeof(Pixel));

    for(int i=1;i<=P->y_image_size;i++)
    {
      Part[i-1]=P->Image_Array[1][i];
    }
    P->SentArrays[1]=Part;
  }
  else if(x_index==-1 && y_index==1)        //NE
  {
    *size=1;
    Part=malloc(1*sizeof(Pixel));
    Part[0]=P->Image_Array[1][P->y_image_size];
    P->SentArrays[2]=Part;
  }
  else if(x_index==0 && y_index==-1)      //W
  {
    *size=P->x_image_size;
    Part=malloc(P->x_image_size*sizeof(Pixel));

    for(int i=1;i<=P->x_image_size;i++)
    {
      Part[i-1]=P->Image_Array[i][1];
    }
    P->SentArrays[3]=Part;
  }
  else if(x_index==0 && y_index==1)   //E
  {
    *size=P->x_image_size;
    Part=malloc(P->x_image_size*sizeof(Pixel));

    for(int i=1;i<=P->x_image_size;i++)
    {
      Part[i-1]=P->Image_Array[i][P->y_image_size];
    }
      P->SentArrays[4]=Part;
  }
  else if(x_index==1 && y_index==-1)  //SW
  {
    *size=1;
    Part=malloc(1*sizeof(Pixel));
    Part[0]=P->Image_Array[P->x_image_size][1];
    P->SentArrays[5]=Part;
  }
  else if(x_index==1 && y_index==0) //S
  {
    *size=P->y_image_size;
    Part=malloc(P->y_image_size*sizeof(Pixel));

    for(int i=1;i<=P->y_image_size;i++)
    {
      Part[i-1]=P->Image_Array[P->x_image_size][i];
    }
    P->SentArrays[6]=Part;
  }
  else if(x_index==1 && y_index==1)     //SE
  {
    *size=1;
    Part=malloc(1*sizeof(Pixel));
    Part[0]=P->Image_Array[P->x_image_size][P->y_image_size];
    P->SentArrays[7]=Part;
  }
  return Part;
}


static inline void Send_Arrays(Grey_Process *P)
{

  int i,j,index=0,neig_index=0,size;
  Pixel* part;


  for(i=-1;i<=1;i++)
  {
    for(j=-1;j<=1;j++)
    {
      if(i!=0 || j!=0)
      {
        if(P->neighbour_proc_ranks[index]!=-1)
        {

          part=GetImagePart(i,j,P,&size);

          MPI_Isend(part,size,MPI_UNSIGNED_CHAR,P->neighbour_proc_ranks[index],P->rank,MPI_COMM_WORLD,&P->Send_r[neig_index]);
          neig_index++;
        }
        index++;
      }
    }
  }

}

static inline void Receive_Arrays(Grey_Process* P)
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

          MPI_Irecv(&P->Received[array_index],size,MPI_UNSIGNED_CHAR,P->neighbour_proc_ranks[index],P->neighbour_proc_ranks[index],MPI_COMM_WORLD,&P->Receiv_r[neig_index]);

          neig_index++;
        }
        array_index+=size;
        index++;
        }
      }
  }

}


#endif
