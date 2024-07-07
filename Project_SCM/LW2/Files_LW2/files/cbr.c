#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

char filename[50] = "Terse_DieFirma_H261_64kbps.dat";

int number_inputs = 0;

int main(int argc, char *argv[]){

    int right = 0;
    long int counterBufDataRecv = 0;
    float bufferInput = 0;
    float bufferOutput = 0;
    int bufferInputStart = 0;
    int bufferOutputStart = 0;

    char instr_cmd[50] = "python ";

    char python_filename[15] = "graphs.py";

    strcat(instr_cmd, python_filename);


    long int data[100000];


    if(argc == 4){
        printf("Right number of input arguments\n");
        right = 1;
    }else{
        if (argc < 4){
            printf("Too few input arguments\n");
        }else{
            if(argc > 4){
                printf("Too many imput arguments\n");
            }
        }
    }

    if(right == 1){      // 0
        strcpy(filename, argv[1]);

        float bit_rate = atof(argv[2]);
        int buffer_length = atoi(argv[3]);

     //   int buffer_length = 20000;
        int decis_occup = buffer_length/2;
     //   float bit_rate = 63925;
        int fps = 25;
        int bits_per_pix = 8;

        float bit_rate_iter = bit_rate/(fps*bits_per_pix);

        char bitRateStr[12]= "";
        sprintf(bitRateStr, "%d",bit_rate_iter);

        int count = 0;
        int countx = 0;

        char bufLenStr[12]= "";
        sprintf(bufLenStr, "%d",buffer_length);


        if (strstr(filename, ".dat") != NULL){
       //   printf("Nice");

            FILE *file;
            file =fopen(filename,"r");

            FILE * fBufferData;
            fBufferData= fopen("buffer_management.csv", "w");
            fprintf(fBufferData, "Time, StartPointBuffer, EndPointBuffer\n");

			if(!file)
			{
					perror("Could not open the file");
					return -1;
			}

            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) != NULL) {
                printf("Current working dir: %s\n", cwd);
            } else {
                perror("getcwd() error");
                return 1;
            }

            while (!feof(file)){

                if(feof(file)){
                    printf("\nEnd of file--------------------------------\n");
                    break;
                }

                if(count < 100000){

                    fscanf(file, "%d", &(data[count]));

                    countx++;

                    bufferInput += data[count];

                    if(bufferInput >= decis_occup){
                        if(bufferInput > 0){
                            if(bufferInput <= buffer_length){
                                bufferOutput += bit_rate_iter;
                                bufferInput -= bit_rate_iter;
                            }else{
                               printf("Warning: Input buffer Overflow has been detected at frame %d\n", count);
                               bufferInput = buffer_length;  // crop buffer data that makes the buffer size bigger than buffer_length
                            }
                        }else{
                            printf("Warning: Input buffer Underflow has been detected at frame %d\n", count);
                            bufferInput = 0;   //set to the minimum
                        }
                    }

                    if(bufferOutput >= decis_occup){
                        counterBufDataRecv++;
                        if(bufferOutput > 0){
                            bufferOutput -= data[counterBufDataRecv];  // Crop buffer data until the buffer value becomes zero or negative
                            if(bufferOutput > buffer_length){
                                printf("Warning: Output buffer Overflow has been detected at frame %d\n", counterBufDataRecv);
                                bufferOutput = buffer_length;  // crop buffer data that makes the buffer size bigger than buffer_length
                            }
                        }else{
                            printf("Warning: Output buffer Underflow has been detected at frame %d\n", counterBufDataRecv);
                            bufferOutput = 0;    //set to the minimum
                        }
                    }

                    if(data[count] == 0){
                        break;
                    }

                    fprintf(fBufferData, "%f, %f, %f\n", 0.04*count, bufferInput, bufferOutput);

                    count++;
                }
            }

           number_inputs = countx;

           fclose(file);
	     fclose(fBufferData);

           strcat(instr_cmd, " ");
           strcat(instr_cmd, filename);
           strcat(instr_cmd, " ");
           strcat(instr_cmd, bitRateStr);
           strcat(instr_cmd, " ");
           strcat(instr_cmd, bufLenStr);
           strcat(instr_cmd, " ");
           strcat(instr_cmd, cwd);

           printf("\nGoing to generate output graphs, for: \n %s\n", instr_cmd);

           system(instr_cmd);


        }
    }
}
