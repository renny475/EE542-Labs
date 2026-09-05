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

#define PROGRESS_INTERVAL (10ULL * 1024 * 1024)

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

static long timespec_diff_ms(const struct timespec *a, const struct timespec *b)
{
    return (a->tv_sec - b->tv_sec) * 1000L + (a->tv_nsec - b->tv_nsec) / 1000000L;
}

static long timespec_diff_ns(const struct timespec *a, const struct timespec *b)
{
    return (a->tv_sec - b->tv_sec) * 1000000000L + (a->tv_nsec - b->tv_nsec);
}

/* Rate-limits NACK sends the same way client.c's pace_send() rate-
 * limits data sends: batch bytes freely until NACK_PACE_BATCH_BYTES is
 * reached, then one sleep brings the batch back to NACK_PACE_TARGET_MBPS
 * before continuing. Needed now that a single sweep can address the
 * whole window's worth of gaps (NACK_SWEEP_CAP == WINDOW_SIZE, see
 * common.h) - without pacing, a dense sweep under heavy loss would
 * fire hundreds or thousands of NACKs in a tight loop, blowing through
 * the tc tbf burst allowance just like an unpaced data burst would. */
static void pace_nack_send(void)
{
    static struct timespec batch_start = {0, 0};
    static size_t batch_bytes = 0;
    struct timespec now;
    const size_t wire_bytes = sizeof(packet_header_t) + PACE_OVERHEAD_BYTES;

    clock_gettime(CLOCK_MONOTONIC, &now);

    if (batch_start.tv_sec == 0 && batch_start.tv_nsec == 0)
        batch_start = now;

    batch_bytes += wire_bytes;

    if (batch_bytes >= NACK_PACE_BATCH_BYTES) {
        long elapsed_ns = timespec_diff_ns(&now, &batch_start);
        long target_ns = (long)((double)batch_bytes * 8.0
                                 / ((double)NACK_PACE_TARGET_MBPS * 1000000.0) * 1e9);
        long wait_ns = target_ns - elapsed_ns;

        if (wait_ns > 0) {
            struct timespec sleep_ts;

            sleep_ts.tv_sec = wait_ns / 1000000000L;
            sleep_ts.tv_nsec = wait_ns % 1000000000L;
            nanosleep(&sleep_ts, NULL);
        }

        clock_gettime(CLOCK_MONOTONIC, &batch_start);
        batch_bytes = 0;
    }
}

static void send_ctrl(int sockfd, struct sockaddr_in *addr, socklen_t len,
                       uint16_t flags, uint32_t seq)
{
    packet_t pkt;

    pkt.header.seq_num = seq;
    pkt.header.ack_num = 0;
    pkt.header.flags = flags;
    pkt.header.data_len = 0;
    packet_set_checksum(&pkt);

    sendto(sockfd, &pkt, sizeof(packet_header_t), 0, (struct sockaddr *)addr, len);
}

/* Diagnostic counter only - total NACKs sent across every sweep, used
 * to compare against the number of distinct loss events (see the
 * stats printed at the end of main()). No effect on protocol behavior. */
static uint64_t stat_nacks_sent = 0;

/* pending_since[slot] doubles as both "when was this gap first seen"
 * (the NACK_MIN_AGE_MS grace period before the first NACK) and, once a
 * NACK has actually been sent, "when did we last NACK this gap" - it
 * gets reset to now() right after sending, so the same gap won't be
 * NACKed again until another full NACK_MIN_AGE_MS has passed. Without
 * this, under a real RTT close to or above NACK_MIN_AGE_MS (Case 2's
 * 200ms RTT vs. a 300ms grace period), every sweep after the first
 * NACK keeps re-NACKing the same still-outstanding gap every
 * NACK_THROTTLE_MS (200ms) - the resend can't possibly have arrived
 * yet, so each redundant NACK just triggers another redundant
 * retransmit. Observed on the real VMs: 351,420 NACKs sent and ~278k
 * duplicate packets received for a 1M-packet Case 2 transfer, driving
 * throughput down to 15Mbit/s - both numbers are far above what a 20%
 * loss rate alone would require. */
static void nack_window_gaps(int sockfd, struct sockaddr_in *addr, socklen_t len,
                              const uint8_t *received, struct timespec *pending_since,
                              uint32_t rcv_base, uint32_t window_end,
                              struct timespec *last_sweep)
{
    struct timespec now;
    uint32_t seq;
    uint32_t sent = 0;

    clock_gettime(CLOCK_MONOTONIC, &now);
    if (timespec_diff_ms(&now, last_sweep) < NACK_THROTTLE_MS)
        return;

    for (seq = rcv_base; seq < window_end && sent < NACK_SWEEP_CAP; seq++) {
        uint32_t slot = seq % WINDOW_SIZE;

        if (received[slot])
            continue;

        if (pending_since[slot].tv_sec == 0 && pending_since[slot].tv_nsec == 0) {
            pending_since[slot] = now;
        } else if (timespec_diff_ms(&now, &pending_since[slot]) >= NACK_MIN_AGE_MS) {
            pace_nack_send();
            send_ctrl(sockfd, addr, len, FLAG_NACK, seq);
            pending_since[slot] = now;
            sent++;
            stat_nacks_sent++;
        }
    }

    *last_sweep = now;
}

static uint32_t nack_sweep_end(uint32_t rcv_base, uint32_t highest_seq_seen,
                                int eof_received, uint32_t eof_seq)
{
    uint32_t window_cap, reachable;

    if (eof_received)
        return eof_seq;

    window_cap = rcv_base + WINDOW_SIZE;
    reachable = highest_seq_seen + 1;

    return reachable < window_cap ? reachable : window_cap;
}

int main(int argc, char *argv[])
{
    int sockfd, portno, outfd;
    ssize_t n;
    struct sockaddr_in serv_addr, sender_addr;
    socklen_t sender_len;
    packet_t pkt;
    struct timespec start_time, end_time, finish_realtime;
    struct timespec last_sweep = {0, 0};
    uint64_t bytes_received = 0;
    uint64_t last_reported = 0;
    int timer_started = 0;
    size_t payload_size;

    uint8_t *received; /* WINDOW_SIZE slots, indexed by seq % WINDOW_SIZE */
    struct timespec *pending_since; /* WINDOW_SIZE slots, parallel to received[] */
    uint32_t rcv_base = 0;
    uint32_t highest_seq_seen = 0;
    uint32_t eof_seq = 0;
    int eof_received = 0;
    int retries = 0;
    struct timeval tv;

    /* Diagnostic counters only - no effect on protocol behavior. */
    uint64_t stat_new_data = 0, stat_dup_below_base = 0, stat_dup_in_window = 0;

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

    received = calloc(WINDOW_SIZE, 1);
    pending_since = calloc(WINDOW_SIZE, sizeof(struct timespec));
    if (!received || !pending_since)
        error("ERROR allocating window tracking table");

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        error("ERROR opening UDP socket");

    /* See the matching comment in client.c: a generous receive buffer
     * keeps a burst of incoming data packets from being silently
     * dropped by the kernel before this process gets to read them. */
    {
        int rcvbuf = 4 * 1024 * 1024;

        setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    }

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR binding UDP socket");

    printf("UDP receiver listening on port %d, saving to %s (chunk_size=%zu, window=%d)\n",
           portno, argv[2], payload_size, WINDOW_SIZE);
    printf("Waiting for file data from sender...\n");

    sender_len = sizeof(sender_addr);

    tv.tv_sec = TAIL_TIMEOUT_MS / 1000;
    tv.tv_usec = (TAIL_TIMEOUT_MS % 1000) * 1000;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
        error("ERROR setting receive timeout");

    while (!(eof_received && rcv_base >= eof_seq) && retries < TAIL_MAX_RETRIES) {
        n = recvfrom(sockfd, &pkt, sizeof(pkt), 0,
                     (struct sockaddr *)&sender_addr, &sender_len);

        if (n < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                retries++;
                if (eof_received)
                    send_ctrl(sockfd, &sender_addr, sender_len, FLAG_ACK, eof_seq);
                nack_window_gaps(sockfd, &sender_addr, sender_len, received, pending_since,
                                  rcv_base,
                                  nack_sweep_end(rcv_base, highest_seq_seen, eof_received, eof_seq),
                                  &last_sweep);
                continue;
            }
            error("ERROR receiving UDP data");
        }

        retries = 0;

        if (!packet_verify_checksum(&pkt, (size_t)n))
            continue;

        if (!timer_started) {
            clock_gettime(CLOCK_MONOTONIC, &start_time);
            timer_started = 1;
        }

        if (pkt.header.flags == FLAG_EOF) {
            eof_received = 1;
            eof_seq = pkt.header.seq_num;
            send_ctrl(sockfd, &sender_addr, sender_len, FLAG_ACK, eof_seq);
            continue;
        }

        {
            uint32_t seq = pkt.header.seq_num;

            if (seq > highest_seq_seen)
                highest_seq_seen = seq;

            if (seq < rcv_base) {
                /* Already delivered; the ACK for it must have been
                 * lost, so re-ACK to stop the sender retransmitting it
                 * forever. */
                send_ctrl(sockfd, &sender_addr, sender_len, FLAG_ACK, seq);
                stat_dup_below_base++;
            } else if (seq < rcv_base + WINDOW_SIZE) {
                uint32_t slot = seq % WINDOW_SIZE;

                if (!received[slot]) {
                    if (pwrite(outfd, pkt.data, pkt.header.data_len,
                               (off_t)seq * payload_size) < 0)
                        error("ERROR writing received data to output file");

                    received[slot] = 1;
                    bytes_received += pkt.header.data_len;
                    stat_new_data++;
                } else {
                    stat_dup_in_window++;
                }

                send_ctrl(sockfd, &sender_addr, sender_len, FLAG_ACK, seq);

                while (received[rcv_base % WINDOW_SIZE]) {
                    uint32_t freed_slot = rcv_base % WINDOW_SIZE;

                    received[freed_slot] = 0;
                    pending_since[freed_slot].tv_sec = 0;
                    pending_since[freed_slot].tv_nsec = 0;
                    rcv_base++;
                }
            }
            /* seq >= rcv_base + WINDOW_SIZE would mean the sender sent
             * ahead of the agreed window - shouldn't happen since it
             * respects WINDOW_SIZE too, so just drop it defensively. */
        }

        nack_window_gaps(sockfd, &sender_addr, sender_len, received, pending_since,
                          rcv_base,
                          nack_sweep_end(rcv_base, highest_seq_seen, eof_received, eof_seq),
                          &last_sweep);

        if (bytes_received - last_reported >= PROGRESS_INTERVAL) {
            printf("Progress: %.2f MiB received\n", bytes_received / (1024.0 * 1024.0));
            fflush(stdout);
            last_reported = bytes_received;
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

        if (!(eof_received && rcv_base >= eof_seq))
            printf("WARNING: gave up with %u packet(s) still missing (first missing seq=%u)\n",
                   eof_seq - rcv_base, rcv_base);

        printf("Stats: new_data=%llu dup_below_base(ack_lost_recovery)=%llu dup_in_window=%llu nacks_sent=%llu\n",
               (unsigned long long)stat_new_data, (unsigned long long)stat_dup_below_base,
               (unsigned long long)stat_dup_in_window, (unsigned long long)stat_nacks_sent);
    } else {
        printf("No file data was received.\n");
    }

    fsync(outfd);
    close(outfd);
    free(received);
    free(pending_since);
    close(sockfd);
    return 0;
}
