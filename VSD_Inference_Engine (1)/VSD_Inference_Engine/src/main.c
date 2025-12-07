#include "uart.h"
#include <stdio.h>
#include <limits.h>
#include "model.h"
#include "inference.h"

#define IMAGE_SIZE 400
#define IMAGE_RES 20

int8_t activations1[L1_incoming_weights];
int32_t output1[L1_outgoing_weights];

int8_t activations2[L2_incoming_weights];
int32_t output2[L2_outgoing_weights];

int8_t activations3[L3_incoming_weights];
int32_t output3[L3_outgoing_weights];

int8_t activations4[L4_incoming_weights];
int32_t output4[L4_outgoing_weights];

uint8_t rx_buffer[IMAGE_SIZE];

void delay_ms(uint32_t ms) {
    for (volatile uint32_t i = 0; i < ms * 4000; i++) {}
}

int BitMnistInference(uint8_t* image_data) {
    
    for (int i = 0; i < L1_incoming_weights; i++) {
        activations1[i] = 128 - image_data[i];
    }

    // Process the layers
    processfclayer(activations1, L1_weights, L1_bitperweight, L1_incoming_weights, L1_outgoing_weights, output1);
    ReLUNorm(output1, activations2, L1_outgoing_weights);

    processfclayer(activations2, L2_weights, L2_bitperweight, L2_incoming_weights, L2_outgoing_weights, output2);
    ReLUNorm(output2, activations3, L2_outgoing_weights);

    processfclayer(activations3, L3_weights, L3_bitperweight, L3_incoming_weights, L3_outgoing_weights, output3);
    ReLUNorm(output3, activations4, L3_outgoing_weights);

    processfclayer(activations4, L4_weights, L4_bitperweight, L4_incoming_weights, L4_outgoing_weights, output4);

    // get the predicted digit
    int32_t max_val = -INT32_MAX;
    int prediction = 0;
    for (int i = 0; i < L4_outgoing_weights; i++) {
        if (output4[i] > max_val) {
            max_val = output4[i];
            prediction = i;
        }
    }

    return prediction;
}

int main(void) {
    uart_init(115200); // same baud rate as transmitter

    while(1) {
        // Get bytes from python transmitter
        printf("\n-----------------------------------\n");
        printf("Board is ready. Waiting for %d bytes from Python script...\n", IMAGE_SIZE);
        printf("(Close this monitor now and run the Python script)\n");
        fflush(stdout);

        for (int i = 0; i < IMAGE_SIZE; i++) {
            rx_buffer[i] = uart_get_char();
        }

        delay_ms(100);

        while(1) {
            printf("\n--- Data Received Successfully ---\n");
            printf("Type 'd' and press Enter to display the stored data.\n");
            printf("> ");
            fflush(stdout);

            char command_buffer[10];
            int i = 0;
            while(i < 9) {
                char c = uart_get_char();
                printf("%c", c);
                fflush(stdout); // to repeat typed character to display
                if (c == '\n' || c == '\r') {
                    break;
                }
                command_buffer[i++] = c;
            }
            command_buffer[i] = '\0';

            if (command_buffer[0] == 'd' && command_buffer[1] == '\0') {
                printf("\n--- Displaying %d bytes of stored data ---\n", IMAGE_SIZE);
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    char pixel_char;
                    if (rx_buffer[j] > 128) {
                        pixel_char = ' ';
                    } else {
                        pixel_char = 'O';
                    }
                    printf("%c ", pixel_char);
                    if ((j + 1) % IMAGE_RES == 0) {
                        printf("\n");
                    }
                }

                printf("\n--- Processing Image ---\n");
                fflush(stdout);
                
                int predicted_digit = BitMnistInference(rx_buffer);

                printf("Inference complete. Predicted Digit: %d\n", predicted_digit);
                fflush(stdout);
                
                printf("\nData displayed. The board will now restart the cycle.\n");
                fflush(stdout);
                delay_ms(2000);
                
                break;
            } else {
                printf("\nUnknown command. Please type 'd'.\n");
            }
        }
    }
}