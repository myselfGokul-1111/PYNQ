#include "cnn_top.h"

// -------------------------------
// 3x3 Convolution Function (Unrolled)
// -------------------------------
void conv3x3(
    data_t in_channels[][IMG_SIZE][IMG_SIZE],
    data_t out_channels[][IMG_SIZE][IMG_SIZE],
    const weight_t weight[][3][3],
    const weight_t bias[],
    int in_ch, int out_ch
) {
    LOOP_OUT: for(int oc=0; oc<out_ch; oc++){
        LOOP_ROW: for(int i=0; i<IMG_SIZE; i++){
            LOOP_COL: for(int j=0; j<IMG_SIZE; j++){
#pragma HLS PIPELINE II=1
                data_t sum = 0;
                LOOP_IN: for(int ic=0; ic<in_ch; ic++){
                    LOOP_KR: for(int kr=-1; kr<=1; kr++){
                        LOOP_KC: for(int kc=-1; kc<=1; kc++){
                            int r = i+kr;
                            int c = j+kc;
                            data_t val = (r>=0 && r<IMG_SIZE && c>=0 && c<IMG_SIZE) ? in_channels[ic][r][c] : 0;
                            sum += val * weight[oc*in_ch + ic][kr+1][kc+1];
                        }
                    }
                }
                out_channels[oc][i][j] = sum + bias[oc];
            }
        }
    }
}

// -------------------------------
// MaxPool 2x2
// -------------------------------
void maxpool2x2(data_t in[][IMG_SIZE][IMG_SIZE], data_t out[][IMG_SIZE/2][IMG_SIZE/2], int ch){
    LOOP_CH: for(int c=0; c<ch; c++){
        LOOP_ROW: for(int i=0; i<IMG_SIZE; i+=2){
            LOOP_COL: for(int j=0; j<IMG_SIZE; j+=2){
#pragma HLS PIPELINE II=1
                data_t m = in[c][i][j];
                if(in[c][i][j+1] > m) m = in[c][i][j+1];
                if(in[c][i+1][j] > m) m = in[c][i+1][j];
                if(in[c][i+1][j+1] > m) m = in[c][i+1][j+1];
                out[c][i/2][j/2] = m;
            }
        }
    }
}

// -------------------------------
// Fully Connected Layer
// -------------------------------
void fc_layer(data_t in[], data_t out[], const weight_t weight[][64], const weight_t bias[], int in_size, int out_size){
    LOOP_OUT: for(int o=0; o<out_size; o++){
#pragma HLS PIPELINE II=1
        data_t sum = 0;
        LOOP_IN: for(int i=0; i<in_size; i++){
            sum += in[i] * weight[i][o];
        }
        out[o] = sum + bias[o];
    }
}

// -------------------------------
// CNN TOP
// -------------------------------
void cnn_top(data_t input[3][IMG_SIZE][IMG_SIZE], data_t output[NUM_CLASSES]){
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=OUTPUT
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Layer buffers
    static data_t conv1_out[16][IMG_SIZE][IMG_SIZE];
    static data_t conv2_out[32][IMG_SIZE/2][IMG_SIZE/2];
    static data_t conv3_out[32][IMG_SIZE/4][IMG_SIZE/4];
    static data_t fc1_in[32*8*8];
    static data_t fc1_out[64];

    // -------------------
    // Conv1 + ReLU
    // -------------------
    conv3x3(input, conv1_out, (weight_t (*)[3][3])conv1_weight, conv1_bias, 3, 16);
    // ReLU
    RELU1: for(int c=0;c<16;c++) for(int i=0;i<IMG_SIZE;i++) for(int j=0;j<IMG_SIZE;j++)
#pragma HLS PIPELINE II=1
        conv1_out[c][i][j] = conv1_out[c][i][j] > 0 ? conv1_out[c][i][j] : 0;

    // -------------------
    // Conv2 + ReLU + MaxPool
    // -------------------
    conv3x3(conv1_out, conv2_out, (weight_t (*)[3][3])conv2_weight, conv2_bias, 16, 32);
    RELU2: for(int c=0;c<32;c++) for(int i=0;i<IMG_SIZE/2;i++) for(int j=0;j<IMG_SIZE/2;j++)
#pragma HLS PIPELINE II=1
        conv2_out[c][i][j] = conv2_out[c][i][j] > 0 ? conv2_out[c][i][j] : 0;
    static data_t pool2_out[32][IMG_SIZE/4][IMG_SIZE/4];
    maxpool2x2(conv2_out, pool2_out, 32);

    // -------------------
    // Conv3 + ReLU + MaxPool
    // -------------------
    conv3x3(pool2_out, conv3_out, (weight_t (*)[3][3])conv3_weight, conv3_bias, 32, 32);
    RELU3: for(int c=0;c<32;c++) for(int i=0;i<IMG_SIZE/4;i++) for(int j=0;j<IMG_SIZE/4;j++)
#pragma HLS PIPELINE II=1
        conv3_out[c][i][j] = conv3_out[c][i][j] > 0 ? conv3_out[c][i][j] : 0;
    static data_t pool3_out[32][8][8];
    maxpool2x2(conv3_out, pool3_out, 32);

    // -------------------
    // Flatten for FC
    // -------------------
    FC_FLATTEN: for(int c=0;c<32;c++)
        for(int i=0;i<8;i++)
            for(int j=0;j<8;j++)
                fc1_in[c*8*8 + i*8 + j] = pool3_out[c][i][j];

    // -------------------
    // FC1 + ReLU
    // -------------------
    fc_layer(fc1_in, fc1_out, (weight_t (*)[64])fc1_weight, fc1_bias, 32*8*8, 64);
    RELU_FC1: for(int i=0;i<64;i++)
#pragma HLS PIPELINE II=1
        fc1_out[i] = fc1_out[i] > 0 ? fc1_out[i] : 0;

    // -------------------
    // FC2 (Output)
    // -------------------
    fc_layer(fc1_out, output, (weight_t (*)[NUM_CLASSES])fc2_weight, fc2_bias, 64, NUM_CLASSES);
}