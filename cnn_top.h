#ifndef CNN_TOP_H
#define CNN_TOP_H

#include <ap_fixed.h>
#include "conv1_weight.h"
#include "conv1_bias.h"
#include "conv2_weight.h"
#include "conv2_bias.h"
#include "conv3_weight.h"
#include "conv3_bias.h"
#include "fc1_weight.h"
#include "fc1_bias.h"
#include "fc2_weight.h"
#include "fc2_bias.h"

#define IMG_SIZE 32
#define NUM_CLASSES 5

typedef ap_fixed<16,6> data_t;  // Q6.10 fixed-point
typedef ap_fixed<16,6> weight_t;

void cnn_top(data_t input[3][IMG_SIZE][IMG_SIZE], data_t output[NUM_CLASSES]);

#endif