#include <string.h>

#include "common.h"

/* Standard Internet-checksum style algorithm: sum 16-bit big-endian
 * words with end-around carry, then take the one's complement. */
uint16_t compute_checksum(const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t sum = 0;
    size_t i;

    for (i = 0; i + 1 < len; i += 2)
        sum += ((uint32_t)bytes[i] << 8) | bytes[i + 1];

    if (i < len)
        sum += (uint32_t)bytes[i] << 8;

    while (sum >> 16)
        sum = (sum & 0xFFFF) + (sum >> 16);

    return (uint16_t)(~sum);
}

void packet_set_checksum(packet_t *pkt)
{
    size_t len = sizeof(packet_header_t) + pkt->header.data_len;

    pkt->header.checksum_ = 0;
    pkt->header.checksum_ = compute_checksum(pkt, len);
}

/* received_len is the number of bytes actually returned by recvfrom(),
 * used to make sure the header's claimed data_len is not lying about
 * how much payload is really there before we checksum over it. */
int packet_verify_checksum(const packet_t *pkt, size_t received_len)
{
    packet_t tmp;
    uint16_t received_checksum;
    size_t len;

    if (received_len < sizeof(packet_header_t))
        return 0;

    len = sizeof(packet_header_t) + pkt->header.data_len;
    if (len > received_len || len > sizeof(packet_t))
        return 0;

    tmp = *pkt;
    received_checksum = tmp.header.checksum_;
    tmp.header.checksum_ = 0;

    return compute_checksum(&tmp, len) == received_checksum;
}
