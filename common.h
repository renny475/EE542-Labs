#include <stdint.h>

#define MAX_PAYLOAD 1024

typedef struct
{
    /* data */
    uint32_t seq_num;
    uint32_t ack_num;
    uint16_t flags;
    uint16_t data_len;
    uint32_t checksum_;
} packet_header_t;

typedef struct {

    packet_header_t header;
    char data[MAX_PAYLOAD];
    
} packet_t;
