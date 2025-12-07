#include "uart.h"

// VSD SDK Hardware imports
#include "ch32v00x_usart.h"
#include "ch32v00x_gpio.h"
#include "ch32v00x_rcc.h"

/*
 * Initializes the USART1 serial port for BOTH Tx and Rx
 * TX = PD5, RX = PD6
 */
void uart_init(uint32_t baudrate) {
    GPIO_InitTypeDef GPIO_InitStructure = {0};
    USART_InitTypeDef USART_InitStructure = {0};

    // Enable clocks for GPIO Port D and USART1
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOD | RCC_APB2Periph_AFIO, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);

    // Configure USART1 TX (PD5) as an output
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_5;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOD, &GPIO_InitStructure);

    // Configure USART1 RX (PD6) as an input
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOD, &GPIO_InitStructure);

    // Configure USART1
    USART_InitStructure.USART_BaudRate = baudrate;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx; // Enable BOTH
    USART_Init(USART1, &USART_InitStructure);

    // Enable USART1
    USART_Cmd(USART1, ENABLE);
}

/*
 * Checks if a character is available to read
 */
int uart_char_available(void) {
    return (USART_GetFlagStatus(USART1, USART_FLAG_RXNE) != RESET);
}

/*
 * Reads a single character from the serial port
 */
char uart_get_char(void) {
    // Wait until data is received
    while (USART_GetFlagStatus(USART1, USART_FLAG_RXNE) == RESET);
    return (char)USART_ReceiveData(USART1);
}

// Below functions are used in the printf function

/*
 * Sends a single character
 */
void uart_send_char(char c) {
    while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
    USART_SendData(USART1, (uint8_t)c);
    while (USART_GetFlagStatus(USART1, USART_FLAG_TC) == RESET);
}

/*
 * Sends a null-terminated string
 */
void uart_send_string(const char* str) {
    while (*str) {
        uart_send_char(*str++);
    }
}