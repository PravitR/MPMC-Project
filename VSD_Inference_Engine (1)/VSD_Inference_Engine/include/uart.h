#ifndef UART_H
#define UART_H

#include <stdint.h>

void uart_init(uint32_t baudrate);
int uart_char_available(void);
char uart_get_char(void);
void uart_send_char(char c);
void uart_send_string(const char* str);

#endif