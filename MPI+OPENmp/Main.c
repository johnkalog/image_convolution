#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
int num_of_threads;
#include "Grey_Process.h"
#include "Rgb_Process.h"



void argument_manage(int argc,char *argv[],char** grey_or_rgb,char **filename,int *x_all,int *y_all,int *loops){
	if ( argc!=7 ){
		printf("Argcs must be 7\n");
		exit(1);
	}
	if ( strcmp(argv[1],"grey")!=0 && strcmp(argv[1],"rgb")!=0 ){
		printf("First arg must be grey or rgb\n");
		exit(1);
	}
	*grey_or_rgb = strdup(argv[1]);
	*filename = strdup(argv[2]);
	if ( atoi(argv[3])==0 || atoi(argv[4])==0 || atoi(argv[5])==0 || atoi(argv[6])==0 ){
		printf("Args 3,4,5 and 6 must be number\n");
		exit(1);
	}
	*x_all = atoi(argv[4]);
	*y_all = atoi(argv[3]);
	*loops = atoi(argv[5]);
	num_of_threads = atoi(argv[6]);
}

int main(int argc, char *argv[])
{
	int comm_sz,my_rank,x_all,y_all,loops;
	char *filename,*grey_or_rgb;
  double finish,start;

	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	argument_manage(argc,argv,&grey_or_rgb,&filename,&x_all,&y_all,&loops);

  double h[3][3];
  h[0][0] = (double)1/16;
  h[0][1] = (double)2/16;
  h[0][2] = (double)1/16;
  h[1][0] = (double)2/16;
  h[1][1] = (double)4/16;
  h[1][2] = (double)2/16;
  h[2][0] = (double)1/16;
  h[2][1] = (double)2/16;
  h[2][2] = (double)1/16;

	if ( strcmp(grey_or_rgb,"grey")==0 ){
		free(grey_or_rgb);
		Grey_Process P;
	  InitializeGrey_Process(&P,filename,my_rank,x_all,y_all,comm_sz);
		free(filename);
			start=MPI_Wtime();
	  for(int i=0;i<loops;i++)
	  {
	    //SENDX8
	    Send_Arrays(&P);
	    ///ReceiveX8
	    Receive_Arrays(&P);

	    ///////CONVOLUTE INSIDE PIXELS///////////

	    insideFilterApply(&P,h);

	    /////////////////////////////////////////
	    MPI_Waitall(P.neighbours_num,P.Receiv_r, MPI_STATUSES_IGNORE);

	    SetParts(&P);

	    ///////CONVOLUTE OUTSIDE PIXELS///////////

	    outsideFilterApply(&P,h);

	    /////////////////////////////////////////

	    MPI_Waitall(P.neighbours_num,P.Send_r, MPI_STATUSES_IGNORE);
				///////////////////////////////////
			for(int i=0;i<=7;i++)
			{
				if(P.SentArrays[i]!=NULL)
				{
					free(P.SentArrays[i]);
				}
			}
			////////////////////////////////////////

	    P.Image_Array = P.New_Image_Array;
	  }
		finish=MPI_Wtime();
	  Change_Image(&P);


	  printf("%f\n",finish-start);


	  DestroyGrey_Process(&P);
	}
	else{
		free(grey_or_rgb);
		RGB_Process P;
	  InitializeProcess_RGB(&P,filename,my_rank,x_all,y_all,comm_sz);
			free(filename);
			start=MPI_Wtime();
	  for(int i=0;i<loops;i++)
	  {
	    //SENDX8
	    Send_Arrays_RGB(&P);
	    ///ReceiveX8
	    Receive_Arrays_RGB(&P);

	    ///////CONVOLUTE INSIDE PIXELS///////////

	    insideFilterApply_RGB(&P,h);

	    /////////////////////////////////////////
	    MPI_Waitall(P.neighbours_num,P.Receiv_r, MPI_STATUSES_IGNORE);

	    SetParts_RGB(&P);

	    ///////CONVOLUTE OUTSIDE PIXELS///////////

	    outsideFilterApply_RGB(&P,h);

	    /////////////////////////////////////////

	    MPI_Waitall(P.neighbours_num,P.Send_r, MPI_STATUSES_IGNORE);
			//////////////////////////////////////////////
			for(int i=0;i<=7;i++)
			{
				if(P.SentArrays[i]!=NULL)
				{
					free(P.SentArrays[i]);
				}
			}
			///////////////////////////////////////////
			 P.Image_Array = P.New_Image_Array;
	  }
		 finish=MPI_Wtime();
	  Change_Image_RGB(&P);


	  printf("%f\n",finish-start);

	  DestroyRGB_Process(&P);
	}

	MPI_Finalize();

	return 0;

}
