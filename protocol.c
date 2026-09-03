#include "protocol.h"

#include <arpa/inet.h>
#include <string.h>

/*
 * Convert a 64-bit value from host byte order to network byte order.
 *
 * Network byte order is big-endian. Most x86/Linux lab VMs are
 * little-endian, so the two 32-bit halves must be swapped and each
 * half converted with htonl().
 */
uint64_t protocol_htonll(uint64_t value)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    uint32_t low = (uint32_t)(value & 0xFFFFFFFFULL);
    uint32_t high = (uint32_t)(value >> 32);

    return ((uint64_t)htonl(low) << 32) | htonl(high);
#else
    return value;
#endif
}

/*
 * Network-to-host conversion is the same operation as host-to-network
 * conversion for a byte-order swap.
 */
uint64_t protocol_ntohll(uint64_t value)
{
    return protocol_htonll(value);
}

/*
 * Build a normal data packet:
 *
 * [8-byte network-order file offset][payload bytes]
 */
int protocol_build_data_packet(
    uint8_t *packet,
    size_t packet_capacity,
    uint64_t sequence,
    const uint8_t *payload,
    size_t payload_len,
    size_t *packet_len)
{
    packet_header_t header;

    if (packet == NULL || payload == NULL || packet_len == NULL)
        return -1;

    if (payload_len == 0 || payload_len > PAYLOAD_SIZE)
        return -1;

    if (packet_capacity < PACKET_HEADER_SIZE + payload_len)
        return -1;

    header.sequence = protocol_htonll(sequence);

    memcpy(packet, &header, PACKET_HEADER_SIZE);
    memcpy(packet + PACKET_HEADER_SIZE, payload, payload_len);

    *packet_len = PACKET_HEADER_SIZE + payload_len;

    return 0;
}

/*
 * Build an end-of-stream packet:
 *
 * [8-byte network-order END_SEQUENCE]
 *
 * No payload follows the header.
 */
int protocol_build_end_packet(
    uint8_t *packet,
    size_t packet_capacity,
    size_t *packet_len)
{
    packet_header_t header;

    if (packet == NULL || packet_len == NULL)
        return -1;

    if (packet_capacity < PACKET_HEADER_SIZE)
        return -1;

    header.sequence = protocol_htonll(END_SEQUENCE);

    memcpy(packet, &header, PACKET_HEADER_SIZE);

    *packet_len = PACKET_HEADER_SIZE;

    return 0;
}

/*
 * Parse a received UDP packet.
 *
 * A normal data packet is:
 * [8-byte sequence][1 to PAYLOAD_SIZE data bytes]
 *
 * An end packet is:
 * [8-byte END_SEQUENCE][no data bytes]
 */
int protocol_parse_packet(
    const uint8_t *packet,
    size_t received_len,
    uint64_t *sequence_out,
    const uint8_t **payload_out,
    size_t *payload_len_out)
{
    packet_header_t header;
    uint64_t sequence;
    size_t payload_len;

    if (packet == NULL ||
        sequence_out == NULL ||
        payload_out == NULL ||
        payload_len_out == NULL) {
        return -1;
    }

    if (received_len < PACKET_HEADER_SIZE)
        return -1;

    memcpy(&header, packet, PACKET_HEADER_SIZE);

    sequence = protocol_ntohll(header.sequence);
    payload_len = received_len - PACKET_HEADER_SIZE;

    if (sequence == END_SEQUENCE) {
        if (payload_len != 0)
            return -1;

        *sequence_out = sequence;
        *payload_out = NULL;
        *payload_len_out = 0;

        return 0;
    }

    if (payload_len == 0 || payload_len > PAYLOAD_SIZE)
        return -1;

    *sequence_out = sequence;
    *payload_out = packet + PACKET_HEADER_SIZE;
    *payload_len_out = payload_len;

    return 0;
}