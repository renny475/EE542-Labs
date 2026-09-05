#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "common.h"

/* server.c is the RECEIVER: it listens for a stream from the sender
 * (client.c) and saves it to the path given on the command line. */

#define PROGRESS_INTERVAL (10ULL * 1024 * 1024)

/* Upper bound on how many distinct seq_nums we track for dedup/gap
 * detection, sized off a generous max transfer size / smallest chunk
 * size we support, with headroom. */
#define MAX_TRACKED_PACKETS ((size_t)((4ULL * 1024 * 1024 * 1024) / DEFAULT_PAYLOAD) + 16)


void error(const char *msg)
{
    perror(msg);
    exit(1);
}


static long timespec_diff_ms(const struct timespec *a, const struct timespec *b)
{
    return (a->tv_sec - b->tv_sec) * 1000L + (a->tv_nsec - b->tv_nsec) / 1000000L;
}


static void send_nack(int sockfd, struct sockaddr_in *sender_addr,
                       socklen_t sender_len, uint32_t seq)
{
    packet_t pkt;

    pkt.header.seq_num = seq;
    pkt.header.ack_num = 0;
    pkt.header.flags = FLAG_NACK;
    pkt.header.data_len = 0;
    packet_set_checksum(&pkt);

    sendto(sockfd, &pkt, sizeof(packet_header_t), 0,
           (struct sockaddr *)sender_addr, sender_len);
}


/* Used during the post-EOF tail wait, where no further packets arrive
 * on their own to reveal gaps one at a time: request every outstanding
 * gap in a single sweep so a burst of consecutive losses is resolved
 * in one round-trip instead of one gap per timeout tick. Also used
 * reactively during phase 1 whenever a later packet exposes a gap. */
static void nack_all_gaps(int sockfd, struct sockaddr_in *sender_addr,
                           socklen_t sender_len, const uint8_t *received,
                           uint32_t expected_seq, uint32_t eof_seq)
{
    uint32_t seq;
    uint32_t sent = 0;

    for (seq = expected_seq; seq < eof_seq && seq < MAX_TRACKED_PACKETS && sent < NACK_SWEEP_CAP; seq++) {
        if (!received[seq]) {
            send_nack(sockfd, sender_addr, sender_len, seq);
            sent++;
        }
    }
}


/* Writes a validated data packet to its correct offset in the output
 * file, marks it received, counts its bytes exactly once, and slides
 * expected_seq forward over any run of packets that are now
 * contiguously complete. */
static void record_data_packet(int outfd, const packet_t *pkt, uint8_t *received,
                                uint32_t *expected_seq, uint64_t *bytes_received,
                                size_t payload_size)
{
    uint32_t seq = pkt->header.seq_num;
    off_t offset = (off_t)seq * payload_size;

    if (seq >= MAX_TRACKED_PACKETS)
        return;

    if (!received[seq]) {
        if (pwrite(outfd, pkt->data, pkt->header.data_len, offset) < 0)
            error("ERROR writing received data to output file");

        received[seq] = 1;
        *bytes_received += pkt->header.data_len;
    }

    while (*expected_seq < MAX_TRACKED_PACKETS && received[*expected_seq])
        (*expected_seq)++;
}


/* Repairs many gaps in parallel instead of chasing the single lowest
 * one: globally throttled to once per NACK_THROTTLE_MS (not per-seq),
 * each call sweeps forward from expected_seq and NACKs up to
 * NACK_SWEEP_CAP outstanding gaps at once, bounded by `upper_bound`
 * (the highest seq_num seen so far during phase 1, or eof_seq once
 * known). This is what lets the receiver keep up when loss produces
 * tens of thousands of simultaneous gaps - resolving one gap per RTT
 * cannot scale, resolving a few hundred per RTT can. The sweep size is
 * capped well under the tbf burst allowance so a sweep can't flood the
 * link and self-inflict extra loss. */
static void periodic_nack_sweep(int sockfd, struct sockaddr_in *sender_addr,
                                 socklen_t sender_len, const uint8_t *received,
                                 uint32_t expected_seq, uint32_t upper_bound,
                                 struct timespec *last_sweep_time)
{
    struct timespec now;
    uint32_t seq;
    uint32_t sent = 0;

    clock_gettime(CLOCK_MONOTONIC, &now);

    if (timespec_diff_ms(&now, last_sweep_time) < NACK_THROTTLE_MS)
        return;

    for (seq = expected_seq;
         seq <= upper_bound && seq < MAX_TRACKED_PACKETS && sent < NACK_SWEEP_CAP;
         seq++) {
        if (!received[seq]) {
            send_nack(sockfd, sender_addr, sender_len, seq);
            sent++;
        }
    }

    *last_sweep_time = now;
}


int main(int argc, char *argv[])
{
    int sockfd, portno, outfd;
    ssize_t n;
    struct sockaddr_in serv_addr, sender_addr;
    socklen_t sender_len;
    packet_t pkt;
    struct timespec start_time, end_time, finish_realtime;
    uint64_t bytes_received = 0;
    uint64_t last_reported = 0;
    int timer_started = 0;
    size_t payload_size;

    uint8_t *received;
    uint32_t expected_seq = 0;
    uint32_t highest_seq_seen = 0;
    uint32_t eof_seq = 0;
    int eof_received = 0;
    int have_sender_addr = 0;

    struct timespec last_sweep_time = {0, 0};
    struct timeval tv;

    if (argc < 3) {
        fprintf(stderr, "usage: %s port output_file [chunk_size]\n", argv[0]);
        exit(1);
    }

    portno = atoi(argv[1]);
    payload_size = (argc >= 4) ? (size_t)atoi(argv[3]) : DEFAULT_PAYLOAD;

    if (payload_size == 0 || payload_size > MAX_PAYLOAD) {
        fprintf(stderr, "ERROR: chunk_size must be between 1 and %d\n", MAX_PAYLOAD);
        exit(1);
    }

    outfd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (outfd < 0)
        error("ERROR opening output file");

    received = calloc(MAX_TRACKED_PACKETS, 1);
    if (!received)
        error("ERROR allocating packet tracking table");

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        error("ERROR opening UDP socket");

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR binding UDP socket");

    printf("UDP receiver listening on port %d, saving to %s (chunk_size=%zu)\n",
           portno, argv[2], payload_size);
    printf("Waiting for file data from sender...\n");

    sender_len = sizeof(sender_addr);

    /* CRITICAL FIX: a receive timeout is now set BEFORE phase 1 begins,
     * not only during the phase 2 tail wait. Previously, phase 1's
     * recvfrom() was an unbounded blocking call, and periodic_nack_sweep()
     * was only ever invoked reactively, right after a packet successfully
     * arrived. If traffic stopped entirely for any reason - the EOF
     * packet itself being lost, the sender crashing or exiting on an
     * error path, or a long enough stretch of consecutive losses - the
     * receiver had no mechanism to notice the silence and had nothing
     * that could ever wake it up again: a permanent, unrecoverable hang
     * with progress output frozen at whatever it last reached. Now, a
     * timeout on recvfrom() during phase 1 fires periodic_nack_sweep()
     * on every idle tick (using highest_seq_seen as the sweep bound,
     * the same evidence-based cap already used reactively), so a lost
     * EOF or a stalled sender is proactively chased with fresh NACKs
     * instead of waited on indefinitely. */
    tv.tv_sec = TAIL_TIMEOUT_MS / 1000;
    tv.tv_usec = (TAIL_TIMEOUT_MS % 1000) * 1000;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
        error("ERROR setting receive timeout");

    /* Phase 1: wait for EOF, tracking data packets, counting bytes
     * exactly once each, and NACKing gaps as they're exposed - either
     * reactively (a later packet reveals a gap) or, now, periodically
     * on idle timeouts so the receiver never waits forever on traffic
     * that has stopped arriving. Bounded by TAIL_MAX_RETRIES
     * consecutive idle timeouts with zero data received, matching the
     * same give-up behavior phase 2 already had, so a sender that's
     * truly gone (crashed, network fully down) doesn't hang the
     * receiver forever either. */
    {
        int idle_retries = 0;

        while (!eof_received && idle_retries < TAIL_MAX_RETRIES) {
            n = recvfrom(sockfd, &pkt, sizeof(pkt), 0,
                         (struct sockaddr *)&sender_addr, &sender_len);

            if (n < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    idle_retries++;
                    if (have_sender_addr)
                        periodic_nack_sweep(sockfd, &sender_addr, sender_len, received,
                                             expected_seq, highest_seq_seen, &last_sweep_time);
                    continue;
                }
                error("ERROR receiving UDP data");
            }

            idle_retries = 0;
            have_sender_addr = 1;

            if (!packet_verify_checksum(&pkt, (size_t)n))
                continue;

            if (!timer_started) {
                clock_gettime(CLOCK_MONOTONIC, &start_time);
                timer_started = 1;
            }

            if (pkt.header.flags == FLAG_EOF) {
                eof_received = 1;
                eof_seq = pkt.header.seq_num;
                break;
            }

            if (pkt.header.seq_num > highest_seq_seen)
                highest_seq_seen = pkt.header.seq_num;

            record_data_packet(outfd, &pkt, received, &expected_seq, &bytes_received, payload_size);
            periodic_nack_sweep(sockfd, &sender_addr, sender_len, received,
                                 expected_seq, highest_seq_seen, &last_sweep_time);

            if (bytes_received - last_reported >= PROGRESS_INTERVAL) {
                printf("Progress: %.2f MiB received\n", bytes_received / (1024.0 * 1024.0));
                fflush(stdout);
                last_reported = bytes_received;
            }
        }

        if (!eof_received)
            printf("WARNING: gave up waiting for EOF after %d idle timeouts "
                   "(no sender traffic received)\n", idle_retries);
    }

    /* Phase 2: tail wait. Keep NACKing the outstanding gaps and waiting
     * for retransmits until we catch up to eof_seq or give up after
     * TAIL_MAX_RETRIES consecutive quiet timeouts. Receive timeout is
     * already set from before phase 1. */
    if (eof_received) {
        int retries = 0;

        while (expected_seq < eof_seq && retries < TAIL_MAX_RETRIES) {
            n = recvfrom(sockfd, &pkt, sizeof(pkt), 0,
                         (struct sockaddr *)&sender_addr, &sender_len);

            if (n < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    retries++;
                    nack_all_gaps(sockfd, &sender_addr, sender_len, received,
                                  expected_seq, eof_seq);
                    continue;
                }
                error("ERROR receiving UDP data during tail wait");
            }

            retries = 0;

            if (!packet_verify_checksum(&pkt, (size_t)n))
                continue;

            if (pkt.header.flags == FLAG_EOF)
                continue;

            record_data_packet(outfd, &pkt, received, &expected_seq, &bytes_received, payload_size);
            periodic_nack_sweep(sockfd, &sender_addr, sender_len, received,
                                 expected_seq, eof_seq, &last_sweep_time);
        }
    }

    clock_gettime(CLOCK_REALTIME, &finish_realtime);

    if (timer_started) {
        double elapsed;
        double throughput_mbps;

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        elapsed =
            (end_time.tv_sec - start_time.tv_sec) +
            (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;

        throughput_mbps =
            (bytes_received * 8.0) / (elapsed * 1000000.0);

        printf("\nTransfer complete\n");
        printf("Bytes received: %llu\n",
               (unsigned long long)bytes_received);
        printf("Elapsed time: %.3f seconds\n", elapsed);
        printf("Receive throughput: %.2f Mbit/s\n", throughput_mbps);
        printf("Receiver finish timestamp (epoch): %lld.%09ld\n",
               (long long)finish_realtime.tv_sec, finish_realtime.tv_nsec);

        if (expected_seq < eof_seq)
            printf("WARNING: gave up with %u packet(s) still missing (first missing seq=%u)\n",
                   eof_seq - expected_seq, expected_seq);
    } else {
        printf("No file data was received.\n");
    }

    fsync(outfd);
    close(outfd);
    free(received);
    close(sockfd);
    return 0;
}
