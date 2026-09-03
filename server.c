#define _GNU_SOURCE

#include "protocol.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SOCKET_BUFFER_BYTES (64 * 1024 * 1024)
#define PROGRESS_INTERVAL_BYTES (10ULL * 1024ULL * 1024ULL)

typedef struct {
    int worker_id;
    int cpu_id;
    int output_fd;
    uint16_t listen_port;
    uint64_t start_offset;
    uint64_t end_offset;
    uint64_t expected_file_size;
    uint64_t bytes_received;
    uint64_t packets_received;
    uint64_t invalid_packets;
    int error;
} receiver_context_t;

static atomic_ullong total_bytes_received = 0;
static atomic_ullong next_progress_report = PROGRESS_INTERVAL_BYTES;

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <base-port> <output-file> <expected-file-size-bytes>\n",
            program);
}

static int parse_base_port(const char *text, uint16_t *port_out)
{
    char *end = NULL;
    long value = strtol(text, &end, 10);

    if (text[0] == '\0' ||
        end == NULL ||
        *end != '\0' ||
        value < 1 ||
        value > 65532) {
        return -1;
    }

    *port_out = (uint16_t)value;
    return 0;
}

static int parse_file_size(const char *text, uint64_t *size_out)
{
    char *end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);

    if (errno != 0 ||
        text[0] == '\0' ||
        end == NULL ||
        *end != '\0' ||
        value == 0) {
        return -1;
    }

    *size_out = (uint64_t)value;
    return 0;
}

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

static void update_progress(uint64_t newly_received_bytes)
{
    uint64_t total;
    uint64_t threshold;

    total = atomic_fetch_add(&total_bytes_received, newly_received_bytes) +
            newly_received_bytes;

    threshold = atomic_load(&next_progress_report);

    while (total >= threshold) {
        if (atomic_compare_exchange_weak(&next_progress_report,
                                         &threshold,
                                         threshold +
                                         PROGRESS_INTERVAL_BYTES)) {
            printf("Progress: %.2f MiB received\n",
                   total / (1024.0 * 1024.0));
            fflush(stdout);
            break;
        }
    }
}

static int pin_current_thread_to_cpu(int cpu_id)
{
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    return pthread_setaffinity_np(pthread_self(),
                                  sizeof(cpuset),
                                  &cpuset);
}

static int configure_socket_buffer(int sockfd)
{
    int buffer_size = SOCKET_BUFFER_BYTES;

    if (setsockopt(sockfd,
                   SOL_SOCKET,
                   SO_RCVBUF,
                   &buffer_size,
                   sizeof(buffer_size)) < 0) {
        return -1;
    }

    return 0;
}

static void *receiver_worker(void *argument)
{
    receiver_context_t *ctx = argument;
    int sockfd = -1;
    struct sockaddr_in server_addr;
    uint8_t packet[MAX_PACKET_SIZE];
    int affinity_result;

    affinity_result = pin_current_thread_to_cpu(ctx->cpu_id);
    if (affinity_result != 0) {
        errno = affinity_result;

        fprintf(stderr,
                "Worker %d: pthread_setaffinity_np CPU %d: %s\n",
                ctx->worker_id,
                ctx->cpu_id,
                strerror(errno));

        ctx->error = 1;
        return NULL;
    }

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        fprintf(stderr,
                "Worker %d: socket: %s\n",
                ctx->worker_id,
                strerror(errno));

        ctx->error = 1;
        return NULL;
    }

    if (configure_socket_buffer(sockfd) != 0) {
        fprintf(stderr,
                "Worker %d: setsockopt(SO_RCVBUF): %s\n",
                ctx->worker_id,
                strerror(errno));

        ctx->error = 1;
        close(sockfd);
        return NULL;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(ctx->listen_port);

    if (bind(sockfd,
             (struct sockaddr *)&server_addr,
             sizeof(server_addr)) < 0) {
        fprintf(stderr,
                "Worker %d: bind UDP port %u: %s\n",
                ctx->worker_id,
                ctx->listen_port,
                strerror(errno));

        ctx->error = 1;
        close(sockfd);
        return NULL;
    }

    printf("Worker %d: pinned to CPU %d, listening on UDP port %u, "
           "accepting offsets [%" PRIu64 ", %" PRIu64 ")\n",
           ctx->worker_id,
           ctx->cpu_id,
           ctx->listen_port,
           ctx->start_offset,
           ctx->end_offset);
    fflush(stdout);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        ssize_t received_len;
        uint64_t sequence;
        const uint8_t *payload;
        size_t payload_len;

        received_len = recvfrom(sockfd,
                                packet,
                                sizeof(packet),
                                0,
                                (struct sockaddr *)&client_addr,
                                &client_addr_len);

        if (received_len < 0) {
            if (errno == EINTR) {
                continue;
            }

            fprintf(stderr,
                    "Worker %d: recvfrom: %s\n",
                    ctx->worker_id,
                    strerror(errno));

            ctx->error = 1;
            break;
        }

        if (protocol_parse_packet(packet,
                                  (size_t)received_len,
                                  &sequence,
                                  &payload,
                                  &payload_len) != 0) {
            ctx->invalid_packets++;
            continue;
        }

        /*
         * END_SEQUENCE is valid only as an 8-byte packet with no payload.
         * protocol_parse_packet() has already verified payload_len == 0.
         */
        if (sequence == END_SEQUENCE) {
            printf("Worker %d: received end marker\n", ctx->worker_id);
            fflush(stdout);
            break;
        }

        /*
         * This worker should receive only packets for its assigned
         * contiguous file range.
         */
        if (sequence < ctx->start_offset ||
            sequence >= ctx->end_offset ||
            payload_len == 0 ||
            sequence > ctx->expected_file_size ||
            payload_len > ctx->expected_file_size - sequence ||
            sequence + payload_len > ctx->end_offset) {
            fprintf(stderr,
                    "Worker %d: rejected packet at offset %" PRIu64
                    " with payload length %zu\n",
                    ctx->worker_id,
                    sequence,
                    payload_len);

            ctx->invalid_packets++;
            continue;
        }

        {
            ssize_t bytes_written;

            bytes_written = pwrite(ctx->output_fd,
                                   payload,
                                   payload_len,
                                   (off_t)sequence);

            if (bytes_written < 0) {
                fprintf(stderr,
                        "Worker %d: pwrite at offset %" PRIu64 ": %s\n",
                        ctx->worker_id,
                        sequence,
                        strerror(errno));

                ctx->error = 1;
                break;
            }

            if ((size_t)bytes_written != payload_len) {
                fprintf(stderr,
                        "Worker %d: incomplete pwrite at offset %" PRIu64 "\n",
                        ctx->worker_id,
                        sequence);

                ctx->error = 1;
                break;
            }
        }

        ctx->bytes_received += (uint64_t)payload_len;
        ctx->packets_received++;

        update_progress((uint64_t)payload_len);
    }

    close(sockfd);

    if (!ctx->error) {
        printf("Worker %d complete: %.2f MiB, %" PRIu64
               " packets, %" PRIu64 " invalid packets\n",
               ctx->worker_id,
               ctx->bytes_received / (1024.0 * 1024.0),
               ctx->packets_received,
               ctx->invalid_packets);
        fflush(stdout);
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    const char *output_file;
    uint16_t base_port;
    uint64_t expected_file_size;
    int output_fd;
    pthread_t workers[THREAD_COUNT];
    receiver_context_t contexts[THREAD_COUNT];
    struct timespec start_time;
    struct timespec end_time;
    uint64_t total_received = 0;
    uint64_t total_packets = 0;
    uint64_t total_invalid_packets = 0;
    int any_error = 0;

    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (parse_base_port(argv[1], &base_port) != 0) {
        fprintf(stderr, "Invalid base port: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    output_file = argv[2];

    if (parse_file_size(argv[3], &expected_file_size) != 0) {
        fprintf(stderr, "Invalid expected file size: %s\n", argv[3]);
        return EXIT_FAILURE;
    }

    output_fd = open(output_file,
                     O_CREAT | O_TRUNC | O_WRONLY,
                     0644);

    if (output_fd < 0) {
        fprintf(stderr,
                "open %s: %s\n",
                output_file,
                strerror(errno));
        return EXIT_FAILURE;
    }

    /*
     * Pre-size the output file. Each receiver uses pwrite() to put its
     * payload at the byte offset stored in the packet header.
     */
    if (ftruncate(output_fd, (off_t)expected_file_size) < 0) {
        fprintf(stderr,
                "ftruncate %s: %s\n",
                output_file,
                strerror(errno));

        close(output_fd);
        return EXIT_FAILURE;
    }

    printf("Four-thread UDP server\n");
    printf("Output file: %s\n", output_file);
    printf("Expected file size: %" PRIu64 " bytes (%.2f MiB)\n",
           expected_file_size,
           expected_file_size / (1024.0 * 1024.0));
    printf("Server ports: %u through %u\n",
           base_port,
           (uint16_t)(base_port + THREAD_COUNT - 1));
    printf("Worker count: %d\n", THREAD_COUNT);
    printf("Payload size: %d bytes\n", PAYLOAD_SIZE);
    printf("Receive buffer requested per worker: %d MiB\n",
           SOCKET_BUFFER_BYTES / (1024 * 1024));
    fflush(stdout);

    for (int i = 0; i < THREAD_COUNT; i++) {
        memset(&contexts[i], 0, sizeof(contexts[i]));

        contexts[i].worker_id = i;
        contexts[i].cpu_id = i;
        contexts[i].output_fd = output_fd;
        contexts[i].listen_port = (uint16_t)(base_port + i);
        contexts[i].expected_file_size = expected_file_size;

        contexts[i].start_offset =
            (expected_file_size * (uint64_t)i) / THREAD_COUNT;

        contexts[i].end_offset =
            (expected_file_size * (uint64_t)(i + 1)) / THREAD_COUNT;
    }

    if (clock_gettime(CLOCK_MONOTONIC, &start_time) != 0) {
        fprintf(stderr, "clock_gettime: %s\n", strerror(errno));
        close(output_fd);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < THREAD_COUNT; i++) {
        int result;

        result = pthread_create(&workers[i],
                                NULL,
                                receiver_worker,
                                &contexts[i]);

        if (result != 0) {
            errno = result;

            fprintf(stderr,
                    "pthread_create worker %d: %s\n",
                    i,
                    strerror(errno));

            /*
             * This early exit is acceptable for this first controlled
             * version. In a later version we can add a shared stop flag
             * and socket timeouts for graceful cancellation.
             */
            close(output_fd);
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < THREAD_COUNT; i++) {
        int result;
        uint64_t expected_worker_bytes;

        result = pthread_join(workers[i], NULL);

        if (result != 0) {
            errno = result;

            fprintf(stderr,
                    "pthread_join worker %d: %s\n",
                    i,
                    strerror(errno));

            any_error = 1;
        }

        total_received += contexts[i].bytes_received;
        total_packets += contexts[i].packets_received;
        total_invalid_packets += contexts[i].invalid_packets;

        if (contexts[i].error) {
            any_error = 1;
        }

        expected_worker_bytes =
            contexts[i].end_offset - contexts[i].start_offset;

        if (contexts[i].bytes_received != expected_worker_bytes) {
            fprintf(stderr,
                    "Worker %d byte mismatch: received %" PRIu64
                    ", expected %" PRIu64 "\n",
                    i,
                    contexts[i].bytes_received,
                    expected_worker_bytes);

            any_error = 1;
        }
    }

    if (fsync(output_fd) < 0) {
        fprintf(stderr, "fsync %s: %s\n", output_file, strerror(errno));
        any_error = 1;
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end_time) != 0) {
        fprintf(stderr, "clock_gettime: %s\n", strerror(errno));
        any_error = 1;
    }

    close(output_fd);

    {
        double elapsed = elapsed_seconds(&start_time, &end_time);
        double throughput_mbps = 0.0;

        if (elapsed > 0.0) {
            throughput_mbps =
                (total_received * 8.0) /
                (elapsed * 1000000.0);
        }

        printf("\nServer transfer summary\n");
        printf("Bytes received: %" PRIu64 "\n", total_received);
        printf("Packets received: %" PRIu64 "\n", total_packets);
        printf("Invalid packets: %" PRIu64 "\n", total_invalid_packets);
        printf("Elapsed time: %.3f seconds\n", elapsed);
        printf("Payload throughput: %.2f Mbit/s\n", throughput_mbps);
    }

    if (total_received != expected_file_size) {
        fprintf(stderr,
                "ERROR: total received bytes do not match expected file size\n");
        any_error = 1;
    }

    if (any_error) {
        fprintf(stderr, "Server transfer failed or was incomplete\n");
        return EXIT_FAILURE;
    }

    printf("Server transfer completed successfully\n");
    return EXIT_SUCCESS;
}