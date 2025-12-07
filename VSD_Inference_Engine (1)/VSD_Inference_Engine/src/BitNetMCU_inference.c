#include <stdint.h>
#include <stdio.h>
#include <limits.h>
#include "inference.h"

// ReLU normalization
uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_input) {
    int32_t max_val = -INT32_MAX;
    uint32_t scale; 
    uint32_t shift = 0; 
    int32_t rounding;
    int32_t tmp;

    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    scale = max_val >> 7; // Divide by 128 
    while (scale > 0) {
        shift++;
        scale >>= 1;
    }

    // Adds 0.5 before truncating to prevent nan and zero errors
    if (shift > 0) {
        rounding = 1 << (shift - 1);
    } else {
        rounding = 0;
    }

    // Apply ReLU and Normalize
    for (uint32_t i = 0; i < n_input; i++) {
        // ReLU: If negative, set to 0
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            // Normalize: (Value + Rounding) / 2^shift
            tmp = (input[i] + rounding) >> shift;  

            // Clip to 127 (max 8-bit signed integer)
            if (tmp > 127) { 
                output[i] = 127;
            } else {
                output[i] = (int8_t)tmp;
            }
        }    
    }
    return 0;
}

// performs matrix multiplication for 1-bit weights
void processfclayer(int8_t *activations, const uint32_t *weights, int32_t bits_per_weight, uint32_t n_input, uint32_t n_output, int32_t *output) 
{
    const uint32_t *weight_ptr = weights;
    
    for (uint32_t i = 0; i < n_output; i++) {
        int8_t *activation_ptr = activations;
        int32_t sum = 0;

        // Because we fit 8 weights (4 bits each) into one 32-bit chunk.
        for (uint32_t k = 0; k < n_input; k += 8) {
            
            // Load 32-bit chunk containing 8 weights
            uint32_t weight_chunk = *weight_ptr++;

            // Process the 8 weights inside the chunk
            for (uint32_t j = 0; j < 8; j++) {
                int32_t in = *activation_ptr++;

                // 4 bit decoding
                
                // If 0x80000000 is set, the weight is Negative.
                // In 4bitsym, 1 = Negative, 0 = Positive.
                int32_t sign = (weight_chunk & 0x80000000) ? -1 : 1;

                // Extract Magnitude (Next 3 bits: Bits 30, 29, 28)
                // Shift right by 28 to move them to the bottom and mask with 0x7 (binary 111) to get magnitude.
                int32_t magnitude = (weight_chunk >> 28) & 0x7;

                // Multiply and Accumulate
                // sum += Input * Sign * Magnitude
                sum += in * sign * magnitude;

                // 4. Shift chunk left by 4 to prepare the next weight
                // This moves the next 4-bit weight into the top position.
                weight_chunk <<= 4;
            }
        }
        output[i] = sum;
    }
}