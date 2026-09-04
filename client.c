#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netdb.h>

#include "common.h"

/* client.c is the SENDER, implementing the sending side of a Selective
 * Repeat sliding window: up to WINDOW_SIZE packets may be unacknowledged
 * "in flight" at once, each held in an in-memory buffer (window_buf) so
 * a retransmit (via NACK or RTO timeout) resends the buffered copy
 * instead of re-reading the file. */

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

static uint32_t total_packets_for_size(off_t total_size, size_t payload_size)
{
    return (uint32_t)((total_size + payload_size - 1) / payload_size);
}

static long timespec_diff_ms(const struct timespec *a, const struct timespec *b)
{
    return (a->tv_sec - b->tv_sec) * 1000L + (a->tv_nsec - b->tv_nsec) / 1000000L;
}

/* Reads packet `seq`'s data from disk into buf and sends it. Only used
 * the first time a packet enters the window - retransmits reuse the
 * already-buffered copy (see resend_buffered) instead of touching disk. */
static void load_and_send(int sockfd, struct sockaddr_in *addr, int filefd,
                           packet_t *buf, uint32_t seq, off_t total_size,
                           size_t payload_size)
{
    off_t offset = (off_t)seq * payload_size;
    off_t remaining = total_size - offset;
    ssize_t data_len = (size_t)remaining < payload_size ? remaining : (ssize_t)payload_size;
    ssize_t n;

    n = pread(filefd, buf->data, (size_t)data_len, offset);
    if (n < 0)
        error("ERROR reading input file");

    buf->header.seq_num = seq;
    buf->header.ack_num = 0;
    buf->header.flags = FLAG_DATA;
    buf->header.data_len = (uint16_t)n;
    packet_set_checksum(buf);

    if (sendto(sockfd, buf, sizeof(packet_header_t) + n, 0,
               (struct sockaddr *)addr, sizeof(*addr)) < 0)
        error("ERROR sending UDP packet");
}

static void resend_buffered(int sockfd, struct sockaddr_in *addr, const packet_t *buf)
{
    sendto(sockfd, buf, sizeof(packet_header_t) + buf->header.data_len, 0,
           (struct sockaddr *)addr, sizeof(*addr));
}

static void send_eof(int sockfd, struct sockaddr_in *addr, uint32_t total_packets)
{
    packet_t pkt;

    pkt.header.seq_num = total_packets;
    pkt.header.ack_num = 0;
    pkt.header.flags = FLAG_EOF;
    pkt.header.data_len = 0;
    packet_set_checksum(&pkt);

    sendto(sockfd, &pkt, sizeof(packet_header_t), 0, (struct sockaddr *)addr, sizeof(*addr));
}

int main(int argc, char *argv[])
{
    int sockfd, portno, filefd;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    struct stat st;
    off_t total_size;
    size_t payload_size;
    uint32_t total_packets;
    uint32_t send_base = 0, next_seq = 0;
    packet_t *window_buf;
    uint8_t *acked;
    struct timespec *last_sent;
    struct timespec send_start, now;
    struct timeval tv;
    int retries;

    if (argc < 4) {
        fprintf(stderr, "usage: %s receiver_host port file_to_send [chunk_size]\n", argv[0]);
        exit(1);
    }

    portno = atoi(argv[2]);
    payload_size = (argc >= 5) ? (size_t)atoi(argv[4]) : DEFAULT_PAYLOAD;

    if (payload_size == 0 || payload_size > MAX_PAYLOAD) {
        fprintf(stderr, "ERROR: chunk_size must be between 1 and %d\n", MAX_PAYLOAD);
        exit(1);
    }

    filefd = open(argv[3], O_RDONLY);
    if (filefd < 0)
        error("ERROR opening input file");

    if (fstat(filefd, &st) < 0)
        error("ERROR getting input file size");

    total_size = st.st_size;
    total_packets = total_packets_for_size(total_size, payload_size);

    window_buf = calloc(WINDOW_SIZE, sizeof(packet_t));
    acked = calloc(WINDOW_SIZE, 1);
    last_sent = calloc(WINDOW_SIZE, sizeof(struct timespec));
    if (!window_buf || !acked || !last_sent)
        error("ERROR allocating sliding-window buffers");

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        error("ERROR opening UDP socket");

    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr, "ERROR, no such host\n");
        close(filefd);
        close(sockfd);
        exit(1);
    }

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;

    bcopy((char *)server->h_addr,
          (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);

    serv_addr.sin_port = htons(portno);

    /* A short receive timeout paces the main loop: it services
     * incoming ACK/NACKs and gives a regular tick to check per-slot
     * RTO timeouts, without busy-spinning the CPU. */
    tv.tv_sec = 0;
    tv.tv_usec = 20000; /* 20ms */
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
        error("ERROR setting receive timeout");

    clock_gettime(CLOCK_REALTIME, &send_start);
    printf("Sender start timestamp (epoch): %lld.%09ld\n",
           (long long)send_start.tv_sec, send_start.tv_nsec);
    printf("Sending %s (%lld bytes, chunk_size=%zu, %u packets, window=%d)\n",
           argv[3], (long long)total_size, payload_size, total_packets, WINDOW_SIZE);

    while (send_base < total_packets) {
        /* Fill the window with new packets up to its limit. */
        while (next_seq < send_base + WINDOW_SIZE && next_seq < total_packets) {
            uint32_t slot = next_seq % WINDOW_SIZE;

            load_and_send(sockfd, &serv_addr, filefd, &window_buf[slot], next_seq,
                          total_size, payload_size);
            clock_gettime(CLOCK_MONOTONIC, &last_sent[slot]);
            acked[slot] = 0;
            next_seq++;
        }

        /* Service every ACK/NACK that has already arrived, not just one.
         * The first recvfrom blocks up to the 20ms timeout, pacing the
         * loop when nothing is pending; once anything arrives, drain
         * the rest of the queue non-blockingly before refilling the
         * window. Processing only one reply per iteration would cap
         * throughput to roughly one packet per loop round, regardless
         * of how large WINDOW_SIZE is - a real problem once RTT is
         * non-negligible (unlike on loopback). */
        {
            packet_t pkt;
            ssize_t n;
            int flags = 0;

            for (;;) {
                n = recvfrom(sockfd, &pkt, sizeof(pkt), flags, NULL, NULL);
                flags = MSG_DONTWAIT;

                if (n < 0) {
                    if (errno == EWOULDBLOCK || errno == EAGAIN)
                        break;
                    error("ERROR receiving ACK/NACK");
                }

                if (packet_verify_checksum(&pkt, (size_t)n)) {
                    uint32_t seq = pkt.header.seq_num;

                    if (seq >= send_base && seq < next_seq) {
                        uint32_t slot = seq % WINDOW_SIZE;

                        if (pkt.header.flags == FLAG_ACK) {
                            acked[slot] = 1;
                        } else if (pkt.header.flags == FLAG_NACK) {
                            resend_buffered(sockfd, &serv_addr, &window_buf[slot]);
                            clock_gettime(CLOCK_MONOTONIC, &last_sent[slot]);
                        }
                    }
                }
            }
        }

        /* Slide the window forward over contiguously-acked slots. */
        while (send_base < next_seq && acked[send_base % WINDOW_SIZE])
            send_base++;

        /* RTO fallback: resend anything unacknowledged for too long,
         * in case both the original packet and any NACK for it were
         * lost. */
        clock_gettime(CLOCK_MONOTONIC, &now);
        {
            uint32_t seq;

            for (seq = send_base; seq < next_seq; seq++) {
                uint32_t slot = seq % WINDOW_SIZE;

                if (!acked[slot] && timespec_diff_ms(&now, &last_sent[slot]) >= RTO_MS) {
                    resend_buffered(sockfd, &serv_addr, &window_buf[slot]);
                    last_sent[slot] = now;
                }
            }
        }
    }

    /* All data has been acknowledged. Send EOF and wait for it to be
     * acked too, retrying on timeout until confirmed or we give up. */
    retries = 0;
    while (retries < TAIL_MAX_RETRIES) {
        packet_t pkt;
        ssize_t n;

        send_eof(sockfd, &serv_addr, total_packets);

        tv.tv_sec = TAIL_TIMEOUT_MS / 1000;
        tv.tv_usec = (TAIL_TIMEOUT_MS % 1000) * 1000;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        n = recvfrom(sockfd, &pkt, sizeof(pkt), 0, NULL, NULL);
        if (n >= 0 && packet_verify_checksum(&pkt, (size_t)n) &&
            pkt.header.flags == FLAG_ACK && pkt.header.seq_num == total_packets) {
            break;
        }
        retries++;
    }

    {
        struct timespec send_done;
        double elapsed;

        clock_gettime(CLOCK_REALTIME, &send_done);
        elapsed = (send_done.tv_sec - send_start.tv_sec) +
                  (send_done.tv_nsec - send_start.tv_nsec) / 1000000000.0;

        printf("Sender done timestamp (epoch): %lld.%09ld\n",
               (long long)send_done.tv_sec, send_done.tv_nsec);
        printf("Sender-side elapsed: %.3f seconds\n", elapsed);
    }

    printf("UDP file send completed.\n");

    free(window_buf);
    free(acked);
    free(last_sent);
    close(filefd);
    close(sockfd);

    return 0;
}
