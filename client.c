#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netdb.h>

#include "common.h"

/* client.c is the SENDER: it reads a local file and streams it to the
 * receiver (server.c). */

#define SOCKET_BUFFER_BYTES (64 * 1024 * 1024)

static uint64_t retransmissions = 0;
static uint64_t nack_requests_received = 0;


void error(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}


static uint32_t total_packets_for_size(off_t total_size,
                                       size_t payload_size)
{
    return (uint32_t)((total_size + payload_size - 1) / payload_size);
}


static long timespec_diff_ns(const struct timespec *a, const struct timespec *b)
{
    return (a->tv_sec - b->tv_sec) * 1000000000L + (a->tv_nsec - b->tv_nsec);
}


/* Rate-limits every data send - first transmission AND every
 * retransmit alike - to PACE_TARGET_MBPS. This is required by the
 * lab's tc qdisc: `tbf rate 100mbit latency 0.001ms burst 9015`
 * allows only a ~9KB (~8 packet) instantaneous burst before dropping
 * outright. Without this, two things blow through that budget:
 *   1. The initial send loop, which would otherwise be limited only
 *      by a flat per-packet usleep unrelated to the actual link rate
 *      or burst allowance.
 *   2. drain_pending_nacks(), which can fire up to NACK_SWEEP_CAP
 *      (512) retransmits back-to-back in a single call whenever the
 *      receiver's NACK sweep has batched up that many gaps (routine
 *      at high loss/RTT, e.g. the lab's 20%/200ms case). Sent
 *      unpaced, only ~8 of those 512 retransmits survive the shaper;
 *      the rest are dropped by tbf itself, not by netem - a massive
 *      self-inflicted loss on top of the intentional 20%, and the
 *      dominant reason throughput collapses at high loss/RTT even
 *      though the same code performs fine at low loss/RTT (small
 *      NACK sweeps rarely exceed the burst budget on their own).
 *
 * Paces in batches (PACE_BATCH_BYTES, sized under the ~9015-byte tbf
 * burst allowance) rather than sleeping after every packet: OS/VM
 * schedulers commonly can't honor a sub-200us nanosleep() precisely
 * (they round up, often to 1ms+), so sleeping once per packet would
 * oversleep and silently throttle throughput far below
 * PACE_TARGET_MBPS. Batching amortizes that fixed per-call error over
 * more bytes while still keeping any single batch's burst under the
 * shaper's allowance. */
static void pace_send(size_t wire_bytes)
{
    static struct timespec batch_start = {0, 0};
    static size_t batch_bytes = 0;
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);

    if (batch_start.tv_sec == 0 && batch_start.tv_nsec == 0)
        batch_start = now;

    batch_bytes += wire_bytes;

    if (batch_bytes >= PACE_BATCH_BYTES) {
        long elapsed_ns = timespec_diff_ns(&now, &batch_start);
        long target_ns = (long)((double)batch_bytes * 8.0
                                 / ((double)PACE_TARGET_MBPS * 1000000.0) * 1e9);
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


/*
 * Request larger UDP queues. The Linux sysctl limits must also permit
 * these requested sizes:
 *
 * sudo sysctl -w net.core.wmem_max=67108864
 * sudo sysctl -w net.core.rmem_max=67108864
 */
static void configure_socket_buffers(int sockfd)
{
    int buffer_size = SOCKET_BUFFER_BYTES;

    if (setsockopt(sockfd,
                   SOL_SOCKET,
                   SO_SNDBUF,
                   &buffer_size,
                   sizeof(buffer_size)) < 0) {
        perror("WARNING: setsockopt(SO_SNDBUF)");
    }

    if (setsockopt(sockfd,
                   SOL_SOCKET,
                   SO_RCVBUF,
                   &buffer_size,
                   sizeof(buffer_size)) < 0) {
        perror("WARNING: setsockopt(SO_RCVBUF)");
    }
}


/*
 * Sends or retransmits the packet identified by seq.
 *
 * The packet data is read from disk every time using pread(), so no
 * memory buffer is needed for retransmission history.
 *
 * Every send goes through pace_send() first - see its comment for why
 * this is required for both first transmissions and retransmits.
 */
static void send_data_packet(int sockfd,
                             struct sockaddr_in *addr,
                             int filefd,
                             uint32_t seq,
                             off_t total_size,
                             size_t payload_size)
{
    packet_t pkt;
    off_t offset = (off_t)seq * (off_t)payload_size;
    off_t remaining = total_size - offset;
    ssize_t data_len;
    ssize_t bytes_read;
    ssize_t bytes_sent;

    if (remaining <= 0)
        return;

    data_len = ((size_t)remaining < payload_size)
        ? (ssize_t)remaining
        : (ssize_t)payload_size;

    memset(&pkt, 0, sizeof(pkt));

    bytes_read = pread(filefd,
                       pkt.data,
                       (size_t)data_len,
                       offset);

    if (bytes_read < 0)
        error("ERROR reading input file for (re)transmit");

    if (bytes_read != data_len) {
        fprintf(stderr,
                "ERROR: short read for packet %u "
                "(read %zd of %zd bytes)\n",
                seq,
                bytes_read,
                data_len);
        exit(EXIT_FAILURE);
    }

    pkt.header.seq_num = seq;
    pkt.header.ack_num = 0;
    pkt.header.flags = FLAG_DATA;
    pkt.header.data_len = (uint16_t)bytes_read;

    packet_set_checksum(&pkt);

    pace_send(sizeof(packet_header_t) + (size_t)bytes_read + PACE_OVERHEAD_BYTES);

    bytes_sent = sendto(sockfd,
                        &pkt,
                        sizeof(packet_header_t) + bytes_read,
                        0,
                        (struct sockaddr *)addr,
                        sizeof(*addr));

    if (bytes_sent < 0)
        error("ERROR sending UDP packet");

    if ((size_t)bytes_sent !=
        sizeof(packet_header_t) + (size_t)bytes_read) {
        fprintf(stderr,
                "ERROR: incomplete UDP send for packet %u "
                "(sent %zd of %zu bytes)\n",
                seq,
                bytes_sent,
                sizeof(packet_header_t) + (size_t)bytes_read);
        exit(EXIT_FAILURE);
    }
}


/*
 * Send the EOF marker. Its seq_num contains the number of data packets
 * in the full input file; valid DATA sequence numbers are 0 to
 * total_packets - 1.
 */
static void send_eof(int sockfd,
                     struct sockaddr_in *addr,
                     uint32_t total_packets)
{
    packet_t pkt;
    ssize_t bytes_sent;

    memset(&pkt, 0, sizeof(pkt));

    pkt.header.seq_num = total_packets;
    pkt.header.ack_num = 0;
    pkt.header.flags = FLAG_EOF;
    pkt.header.data_len = 0;

    packet_set_checksum(&pkt);

    bytes_sent = sendto(sockfd,
                        &pkt,
                        sizeof(packet_header_t),
                        0,
                        (struct sockaddr *)addr,
                        sizeof(*addr));

    if (bytes_sent < 0)
        error("ERROR sending EOF packet");

    if ((size_t)bytes_sent != sizeof(packet_header_t)) {
        fprintf(stderr,
                "ERROR: incomplete UDP send for EOF packet\n");
        exit(EXIT_FAILURE);
    }
}


/*
 * Drain NACK packets that are already queued for this sender socket.
 * MSG_DONTWAIT makes this non-blocking, so the normal initial DATA
 * send loop does not pause while waiting for a NACK.
 *
 * Each retransmit still goes through send_data_packet(), which now
 * paces itself via pace_send() - so a call that drains a large batch
 * of NACKs (e.g. a full NACK_SWEEP_CAP=512 sweep from the receiver,
 * routine at high loss/RTT) no longer fires them all in one
 * unthrottled burst that the tc shaper would mostly drop.
 */
static void drain_pending_nacks(int sockfd,
                                struct sockaddr_in *addr,
                                int filefd,
                                off_t total_size,
                                size_t payload_size)
{
    packet_t pkt;
    ssize_t bytes_received;

    while (1) {
        bytes_received = recvfrom(sockfd,
                                  &pkt,
                                  sizeof(pkt),
                                  MSG_DONTWAIT,
                                  NULL,
                                  NULL);

        if (bytes_received < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN)
                break;

            error("ERROR receiving NACK");
        }

        if (!packet_verify_checksum(&pkt, (size_t)bytes_received))
            continue;

        if (pkt.header.flags == FLAG_NACK) {
            nack_requests_received++;
            retransmissions++;

            send_data_packet(sockfd,
                             addr,
                             filefd,
                             pkt.header.seq_num,
                             total_size,
                             payload_size);
        }
    }
}


int main(int argc, char *argv[])
{
    int sockfd;
    int portno;
    int filefd;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    struct stat st;
    off_t total_size;
    size_t payload_size;
    uint32_t total_packets;
    uint32_t seq;
    struct timeval tv;
    int retries;
    struct timespec send_start;

    if (argc < 4) {
        fprintf(stderr,
                "usage: %s receiver_host port file_to_send [chunk_size]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    portno = atoi(argv[2]);

    payload_size = (argc >= 5)
        ? (size_t)atoi(argv[4])
        : DEFAULT_PAYLOAD;

    if (payload_size == 0 || payload_size > MAX_PAYLOAD) {
        fprintf(stderr,
                "ERROR: chunk_size must be between 1 and %d\n",
                MAX_PAYLOAD);
        exit(EXIT_FAILURE);
    }

    filefd = open(argv[3], O_RDONLY);
    if (filefd < 0)
        error("ERROR opening input file");

    if (fstat(filefd, &st) < 0)
        error("ERROR getting input file size");

    total_size = st.st_size;

    if (total_size <= 0) {
        fprintf(stderr, "ERROR: input file must not be empty\n");
        close(filefd);
        exit(EXIT_FAILURE);
    }

    total_packets = total_packets_for_size(total_size, payload_size);

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        error("ERROR opening UDP socket");

    configure_socket_buffers(sockfd);

    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr, "ERROR: no such host: %s\n", argv[1]);
        close(filefd);
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;

    memcpy(&serv_addr.sin_addr.s_addr,
           server->h_addr,
           (size_t)server->h_length);

    serv_addr.sin_port = htons((uint16_t)portno);

    clock_gettime(CLOCK_REALTIME, &send_start);

    printf("Sender start timestamp (epoch): %lld.%09ld\n",
           (long long)send_start.tv_sec,
           send_start.tv_nsec);

    printf("Sending %s (%lld bytes, chunk_size=%zu, %u packets)\n",
           argv[3],
           (long long)total_size,
           payload_size,
           total_packets);

    printf("Pacing target: %d Mbit/s\n", PACE_TARGET_MBPS);

    printf("Requested UDP socket buffers: %d MiB\n",
           SOCKET_BUFFER_BYTES / (1024 * 1024));

    /*
     * Initial transmission phase.
     *
     * Drain already-arrived NACKs after each packet, but do not print
     * every retransmission. Per-packet terminal output slows recovery.
     *
     * No separate usleep() here anymore: pace_send() (called from
     * inside send_data_packet(), for both this loop and any
     * retransmits drained below) is what limits the send rate now,
     * tied to the actual link/shaper budget instead of a fixed
     * microsecond delay unrelated to it.
     */
    for (seq = 0; seq < total_packets; seq++) {
        send_data_packet(sockfd,
                         &serv_addr,
                         filefd,
                         seq,
                         total_size,
                         payload_size);

        drain_pending_nacks(sockfd,
                            &serv_addr,
                            filefd,
                            total_size,
                            payload_size);
    }

    send_eof(sockfd, &serv_addr, total_packets);

    /*
     * Tail phase:
     * Continue receiving NACKs and retransmitting requested data.
     * Re-send EOF after each receive timeout because EOF itself may
     * have been dropped.
     */
    tv.tv_sec = TAIL_TIMEOUT_MS / 1000;
    tv.tv_usec = (TAIL_TIMEOUT_MS % 1000) * 1000;

    if (setsockopt(sockfd,
                   SOL_SOCKET,
                   SO_RCVTIMEO,
                   &tv,
                   sizeof(tv)) < 0) {
        error("ERROR setting receive timeout");
    }

    retries = 0;

    while (retries < TAIL_MAX_RETRIES) {
        packet_t pkt;
        ssize_t bytes_received;

        bytes_received = recvfrom(sockfd,
                                  &pkt,
                                  sizeof(pkt),
                                  0,
                                  NULL,
                                  NULL);

        if (bytes_received < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                retries++;
                send_eof(sockfd, &serv_addr, total_packets);
                continue;
            }

            error("ERROR receiving NACK during tail wait");
        }

        retries = 0;

        if (!packet_verify_checksum(&pkt, (size_t)bytes_received))
            continue;

        if (pkt.header.flags == FLAG_NACK) {
            nack_requests_received++;
            retransmissions++;

            send_data_packet(sockfd,
                             &serv_addr,
                             filefd,
                             pkt.header.seq_num,
                             total_size,
                             payload_size);
        }
    }

    {
        struct timespec send_done;
        double elapsed;

        clock_gettime(CLOCK_REALTIME, &send_done);

        elapsed =
            (send_done.tv_sec - send_start.tv_sec) +
            (send_done.tv_nsec - send_start.tv_nsec) / 1000000000.0;

        printf("Sender done timestamp (epoch): %lld.%09ld\n",
               (long long)send_done.tv_sec,
               send_done.tv_nsec);

        printf("Sender-side elapsed: %.3f seconds\n", elapsed);

        printf("NACK requests received: %llu\n",
               (unsigned long long)nack_requests_received);

        printf("Retransmissions sent: %llu\n",
               (unsigned long long)retransmissions);
    }

    printf("UDP file send completed.\n");

    close(filefd);
    close(sockfd);

    return EXIT_SUCCESS;
}
