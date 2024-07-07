#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <stdbool.h>

char filename[50] = "Verbose_DieHardIII_H261_64kbps.dat";

int main(int argc, char *argv[]){

    long int count = 0;
    int right = 0;
    long int counterBufDataRecv = 0;
    float bufferInput = 0;
    float bufferOutput = 0;
    int bufferInputStart = 0;
    int bufferOutputStart = 0;
    long int data[89980];     // 90000

    float changedBitRate = 0;
    float sumsFrameRateFramSkipping = 0;

    char instr_cmd[50] = "python ";

    char python_filename[15] = "graphs_2.py";

    strcat(instr_cmd, python_filename);

    if(argc == 3){
        printf("Right number of input arguments\n");
        right = 1;
    }else{
        if (argc < 3){
            printf("Too few input arguments\n");
        }else{
            if(argc > 3){
                printf("Too many imput arguments\n");
            }
        }
    }

    if(right == 1){       // 0
        strcpy(filename, argv[1]);
        int buffer_length = atoi(argv[2]);

    //    int buffer_length =  10000;
        int decis_occup = buffer_length/2;

        char bufLenStr[12]= "";
        sprintf(bufLenStr, "%d",buffer_length);

        if (strstr(filename, ".dat") != NULL){

            FILE *file;
            file =fopen(filename,"r");

            FILE * fBufferData;
            fBufferData= fopen("buffer_management_vbr.csv", "w");
            fprintf(fBufferData, "Time, StartPointBuffer, EndPointBuffer\n");

            if(!file){
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

            char spaceBetArgs[] = " ";
            char *arg_ptr = "";

            float frame_length = 0;
            long int time_ms = 0;
            char frame_type[1000];
            int length_line = 200;
            int row_changed = 0;
            char row[length_line];

            for(int i=0; i< length_line; i++){
                row[i] = "";
            }

            int counterForFrameSkipping = 0;

            while(fgets(row, sizeof(row), file) != NULL){

                row_changed = 1;

                if((strchr(row, '#') == NULL) && (strchr(row, '_') == NULL)){            // If line doesn't include any of those characters
                    int counter = 0;
                    char *arg_ptr = strtok(row, spaceBetArgs);

                    time_ms = atoll(arg_ptr);

                    while ((arg_ptr != NULL) && (row_changed == 1)){


                            if(counter == 0){
                                arg_ptr = strtok(NULL, spaceBetArgs);
                                sprintf(frame_type, "%d", *arg_ptr);
//
//                                if((frame_type == 49) || (frame_type >= 50) || (frame_type == 42)){
//                                    printf("Frame type checked\n");
//                                }else{
//                                    printf("Frame type not known\n");
//                                }
                            }

                            if(counter == 1){
                                counter = -1;
                                row_changed = 0;
                                frame_length = atof(arg_ptr);
                            }

                            counter++;
                    }
                }

                if(time_ms > 40*count){
                   long int i = count;
                   time_ms = 40*i;
                   for(;;){
                        i++;

                        if(counterForFrameSkipping >= 10){
                            counterForFrameSkipping = 0;
                            changedBitRate = sumsFrameRateFramSkipping/10;
                            sumsFrameRateFramSkipping = 0;
                        }

                        sumsFrameRateFramSkipping += frame_length;
                        counterForFrameSkipping++;
                   }
                }else{
                    if(counterForFrameSkipping >= 10){
                        counterForFrameSkipping = 0;
                        changedBitRate = sumsFrameRateFramSkipping/10;
                        sumsFrameRateFramSkipping = 0;
                    }

                    data[count] = frame_length;
                    bufferInput += data[count];

                    if(bufferInput >= decis_occup){
                        if(bufferInput > 0){
                            bufferInput -= changedBitRate;
                            bufferOutput += changedBitRate;

                            if(bufferInput > buffer_length){
                                 printf("Detected Input Buffer overflow at frame %d\n", count);      //buffer overflow
                                 bufferInput = buffer_length;
                            }

                        }else{
                            printf("Detected Input Buffer underflow at frame %d\n", count);      //buffer overflow
                            bufferInput = 0;
                        }
                    }

                    if(bufferOutput >= decis_occup){
                        counterBufDataRecv++;
                        if(bufferOutput > 0){
                            bufferOutput -= data[counterBufDataRecv];

                            if(bufferOutput > buffer_length){
                                 printf("Detected Output Buffer overflow at frame %d\n", count);      //buffer overflow
                                 bufferOutput = buffer_length;
                            }

                        }else{
                            printf("Detected Output Buffer underflow at frame %d\n", count);      //buffer overflow
                            bufferOutput = 0;
                        }

                    }

                    fprintf(fBufferData, "%f, %f, %f\n", 0.04*count, bufferInput, bufferOutput);
                }

                sumsFrameRateFramSkipping += frame_length;
                counterForFrameSkipping++;
                count++;

                if(count == 89981){
                    break;
                }
            }

            // Variable bit rate to string
            char bitRateStr[10]="";
            sprintf(bitRateStr, "VBR");

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
