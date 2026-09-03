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

/* client.c is the SENDER: it reads a local file and streams it to the
 * receiver (server.c). */

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

static uint32_t total_packets_for_size(off_t total_size, size_t payload_size)
{
    return (uint32_t)((total_size + payload_size - 1) / payload_size);
}

/* Sends (or resends) the packet for `seq`, always re-reading the data
 * from disk via pread() at seq*payload_size. This means retransmission
 * needs no in-memory buffer of previously sent packets. */
static void send_data_packet(int sockfd, struct sockaddr_in *addr, int filefd,
                              uint32_t seq, off_t total_size, size_t payload_size)
{
    packet_t pkt;
    off_t offset = (off_t)seq * payload_size;
    off_t remaining = total_size - offset;
    ssize_t data_len = (size_t)remaining < payload_size ? remaining : (ssize_t)payload_size;
    ssize_t n;

    if (data_len <= 0)
        return;

    n = pread(filefd, pkt.data, (size_t)data_len, offset);
    if (n < 0)
        error("ERROR reading input file for (re)transmit");

    pkt.header.seq_num = seq;
    pkt.header.ack_num = 0;
    pkt.header.flags = FLAG_DATA;
    pkt.header.data_len = (uint16_t)n;
    packet_set_checksum(&pkt);

    if (sendto(sockfd, &pkt, sizeof(packet_header_t) + n, 0,
               (struct sockaddr *)addr, sizeof(*addr)) < 0)
        error("ERROR sending UDP packet");
}

static void send_eof(int sockfd, struct sockaddr_in *addr, uint32_t total_packets)
{
    packet_t pkt;

    pkt.header.seq_num = total_packets;
    pkt.header.ack_num = 0;
    pkt.header.flags = FLAG_EOF;
    pkt.header.data_len = 0;
    packet_set_checksum(&pkt);

    if (sendto(sockfd, &pkt, sizeof(packet_header_t), 0,
               (struct sockaddr *)addr, sizeof(*addr)) < 0)
        error("ERROR sending EOF packet");
}

/* Drains any NACKs that have already arrived without blocking the
 * send loop, immediately resending the requested packet for each. */
static void drain_pending_nacks(int sockfd, struct sockaddr_in *addr, int filefd,
                                 off_t total_size, size_t payload_size)
{
    packet_t pkt;
    ssize_t n;

    while (1) {
        n = recvfrom(sockfd, &pkt, sizeof(pkt), MSG_DONTWAIT, NULL, NULL);
        if (n < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN)
                break;
            error("ERROR receiving NACK");
        }

        if (!packet_verify_checksum(&pkt, (size_t)n))
            continue;

        if (pkt.header.flags == FLAG_NACK) {
            printf("Resending packet %u due to NACK\n", pkt.header.seq_num);
            send_data_packet(sockfd, addr, filefd, pkt.header.seq_num, total_size, payload_size);
        }
    }
}

int main(int argc, char *argv[])
{
    int sockfd, portno, filefd;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    struct stat st;
    off_t total_size;
    size_t payload_size;
    uint32_t total_packets, seq;
    struct timeval tv;
    int retries;
    struct timespec send_start;

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

    clock_gettime(CLOCK_REALTIME, &send_start);
    printf("Sender start timestamp (epoch): %lld.%09ld\n",
           (long long)send_start.tv_sec, send_start.tv_nsec);
    printf("Sending %s (%lld bytes, chunk_size=%zu, %u packets)\n",
           argv[3], (long long)total_size, payload_size, total_packets);

    for (seq = 0; seq < total_packets; seq++) {
        send_data_packet(sockfd, &serv_addr, filefd, seq, total_size, payload_size);
        drain_pending_nacks(sockfd, &serv_addr, filefd, total_size, payload_size);
        usleep(25);
    }

    send_eof(sockfd, &serv_addr, total_packets);

    /* Tail-wait phase: keep servicing NACKs (and keep re-sending EOF in
     * case it was itself dropped) until the receiver has gone quiet for
     * TAIL_MAX_RETRIES consecutive timeouts. */
    tv.tv_sec = TAIL_TIMEOUT_MS / 1000;
    tv.tv_usec = (TAIL_TIMEOUT_MS % 1000) * 1000;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
        error("ERROR setting receive timeout");

    retries = 0;
    while (retries < TAIL_MAX_RETRIES) {
        packet_t pkt;
        ssize_t n = recvfrom(sockfd, &pkt, sizeof(pkt), 0, NULL, NULL);

        if (n < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                retries++;
                send_eof(sockfd, &serv_addr, total_packets);
                continue;
            }
            error("ERROR receiving NACK during tail wait");
        }

        retries = 0;

        if (!packet_verify_checksum(&pkt, (size_t)n))
            continue;

        if (pkt.header.flags == FLAG_NACK) {
            printf("Resending packet %u due to NACK (tail)\n", pkt.header.seq_num);
            send_data_packet(sockfd, &serv_addr, filefd, pkt.header.seq_num, total_size, payload_size);
        }
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

    close(filefd);
    close(sockfd);

    return 0;
}
