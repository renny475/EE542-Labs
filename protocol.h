#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

/*
 * Transfer configuration shared by client and server.
 *
 * Four sender threads on the client send to four matching receiver
 * threads on the server. Each worker uses its own UDP port:
 *
 * worker 0 -> base port + 0
 * worker 1 -> base port + 1
 * worker 2 -> base port + 2
 * worker 3 -> base port + 3
 */
#define THREAD_COUNT 4

/*
 * Maximum number of file bytes carried in one UDP datagram.
 *
 * UDP datagram size:
 *   8-byte protocol header + 1400-byte payload = 1408 bytes
 *
 * With IPv4 (20 bytes) and UDP (8 bytes) headers, the approximate
 * IP packet size is 1436 bytes, below a 1500-byte Ethernet MTU.
 */
#define PAYLOAD_SIZE 1400

/* Send the end marker multiple times during the no-loss phase. */
#define STREAM_END_REPEAT_COUNT 5

/*
 * Sequence is a full-file byte offset, not a simple packet counter.
 * This special value is reserved to mark the end of one worker stream.
 */
#define END_SEQUENCE UINT64_MAX

/*
 * The complete application-level packet header.
 *
 * sequence:
 *   - Normal data packet: byte offset where payload belongs in output file.
 *   - End packet: END_SEQUENCE and no payload bytes.
 *
 * The field is transmitted in network byte order.
 */
typedef struct __attribute__((packed)) {
    uint64_t sequence;
} packet_header_t;

#define PACKET_HEADER_SIZE ((size_t)sizeof(packet_header_t))
#define MAX_PACKET_SIZE (PACKET_HEADER_SIZE + PAYLOAD_SIZE)

/*
 * Convert unsigned 64-bit integers between host and network byte order.
 * Standard socket headers provide htonl()/ntohl() only for 32-bit values.
 */
uint64_t protocol_htonll(uint64_t value);
uint64_t protocol_ntohll(uint64_t value);

/*
 * Build a normal data datagram:
 *
 * [8-byte network-order sequence/file offset][payload bytes]
 *
 * Returns 0 on success and -1 on invalid arguments or insufficient
 * packet buffer capacity.
 */
int protocol_build_data_packet(
    uint8_t *packet,
    size_t packet_capacity,
    uint64_t sequence,
    const uint8_t *payload,
    size_t payload_len,
    size_t *packet_len);

/*
 * Build an end-of-stream datagram:
 *
 * [8-byte network-order END_SEQUENCE]
 *
 * Returns 0 on success and -1 on invalid arguments or insufficient
 * packet buffer capacity.
 */
int protocol_build_end_packet(
    uint8_t *packet,
    size_t packet_capacity,
    size_t *packet_len);

/*
 * Parse a UDP datagram created by this protocol.
 *
 * On success:
 *   - sequence_out receives the host-order file offset or END_SEQUENCE.
 *   - payload_out points inside packet, immediately after the header.
 *   - payload_len_out receives the number of payload bytes.
 *
 * Valid normal data packets have 1 to PAYLOAD_SIZE payload bytes.
 * A valid end packet has END_SEQUENCE and zero payload bytes.
 *
 * Returns 0 on success, -1 if the received datagram is malformed.
 */
int protocol_parse_packet(
    const uint8_t *packet,
    size_t received_len,
    uint64_t *sequence_out,
    const uint8_t **payload_out,
    size_t *payload_len_out);

#endif /* PROTOCOL_H */