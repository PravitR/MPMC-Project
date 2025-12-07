#ifndef BITNETMCU_INFERENCE_H
#define BITNETMCU_INFERENCE_H

#include <stdint.h>

// Applies ReLU and normalizes 32-bit sums to 8-bit activations
uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_input);

// Performs the Fully Connected Layer calculation
void processfclayer(int8_t *input, const uint32_t *weights, int32_t bits_per_weight, uint32_t incoming_weights, uint32_t outgoing_weights, int32_t *output);

#endif