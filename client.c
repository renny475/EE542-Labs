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

#define SEND_DELAY_US 500
#define PROGRESS_INTERVAL_BYTES (10ULL * 1024ULL * 1024ULL)

typedef struct {
    int worker_id;
    int cpu_id;
    int input_fd;
    uint64_t start_offset;
    uint64_t end_offset;
    const char *server_ip;
    uint16_t server_port;
    uint64_t bytes_sent;
    uint64_t packets_sent;
    int error;
} sender_context_t;

static atomic_ullong total_bytes_sent = 0;
static atomic_ullong next_progress_report = PROGRESS_INTERVAL_BYTES;

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <server-ip> <base-port> <input-file>\n",
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

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

static void update_progress(uint64_t newly_sent_bytes)
{
    uint64_t total;
    uint64_t threshold;

    total = atomic_fetch_add(&total_bytes_sent, newly_sent_bytes) +
            newly_sent_bytes;

    threshold = atomic_load(&next_progress_report);

    while (total >= threshold) {
        if (atomic_compare_exchange_weak(&next_progress_report,
                                         &threshold,
                                         threshold +
                                         PROGRESS_INTERVAL_BYTES)) {
            printf("Progress: %.2f MiB sent\n",
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

static void *sender_worker(void *argument)
{
    sender_context_t *ctx = argument;
    int sockfd = -1;
    struct sockaddr_in server_addr;
    uint8_t payload[PAYLOAD_SIZE];
    uint8_t packet[MAX_PACKET_SIZE];
    uint64_t offset = ctx->start_offset;
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

    printf("Worker %d: pinned to CPU %d, sending offsets "
           "[%" PRIu64 ", %" PRIu64 ") to port %u\n",
           ctx->worker_id,
           ctx->cpu_id,
           ctx->start_offset,
           ctx->end_offset,
           ctx->server_port);
    fflush(stdout);

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        fprintf(stderr,
                "Worker %d: socket: %s\n",
                ctx->worker_id,
                strerror(errno));
        ctx->error = 1;
        return NULL;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(ctx->server_port);

    if (inet_pton(AF_INET, ctx->server_ip, &server_addr.sin_addr) != 1) {
        fprintf(stderr,
                "Worker %d: invalid server IPv4 address: %s\n",
                ctx->worker_id,
                ctx->server_ip);
        ctx->error = 1;
        close(sockfd);
        return NULL;
    }

    while (offset < ctx->end_offset) {
        size_t requested_bytes = PAYLOAD_SIZE;
        size_t packet_len = 0;
        ssize_t bytes_read;
        ssize_t bytes_sent;

        if (ctx->end_offset - offset < requested_bytes) {
            requested_bytes = (size_t)(ctx->end_offset - offset);
        }

        bytes_read = pread(ctx->input_fd,
                           payload,
                           requested_bytes,
                           (off_t)offset);

        if (bytes_read < 0) {
            fprintf(stderr,
                    "Worker %d: pread at offset %" PRIu64 ": %s\n",
                    ctx->worker_id,
                    offset,
                    strerror(errno));
            ctx->error = 1;
            break;
        }

        if (bytes_read == 0) {
            fprintf(stderr,
                    "Worker %d: unexpected end of file at offset %" PRIu64 "\n",
                    ctx->worker_id,
                    offset);
            ctx->error = 1;
            break;
        }

        if (protocol_build_data_packet(packet,
                                       sizeof(packet),
                                       offset,
                                       payload,
                                       (size_t)bytes_read,
                                       &packet_len) != 0) {
            fprintf(stderr,
                    "Worker %d: protocol_build_data_packet failed\n",
                    ctx->worker_id);
            ctx->error = 1;
            break;
        }

        bytes_sent = sendto(sockfd,
                            packet,
                            packet_len,
                            0,
                            (struct sockaddr *)&server_addr,
                            sizeof(server_addr));

        if (bytes_sent < 0) {
            fprintf(stderr,
                    "Worker %d: sendto: %s\n",
                    ctx->worker_id,
                    strerror(errno));
            ctx->error = 1;
            break;
        }

        if ((size_t)bytes_sent != packet_len) {
            fprintf(stderr,
                    "Worker %d: incomplete UDP datagram send\n",
                    ctx->worker_id);
            ctx->error = 1;
            break;
        }

        offset += (uint64_t)bytes_read;
        ctx->bytes_sent += (uint64_t)bytes_read;
        ctx->packets_sent++;

        update_progress((uint64_t)bytes_read);

        usleep(SEND_DELAY_US);
    }

    if (!ctx->error) {
        uint8_t end_packet[PACKET_HEADER_SIZE];
        size_t end_packet_len = 0;

        if (protocol_build_end_packet(end_packet,
                                      sizeof(end_packet),
                                      &end_packet_len) != 0) {
            fprintf(stderr,
                    "Worker %d: protocol_build_end_packet failed\n",
                    ctx->worker_id);
            ctx->error = 1;
        } else {
            for (int attempt = 0;
                 attempt < STREAM_END_REPEAT_COUNT;
                 attempt++) {
                ssize_t bytes_sent;

                bytes_sent = sendto(sockfd,
                                    end_packet,
                                    end_packet_len,
                                    0,
                                    (struct sockaddr *)&server_addr,
                                    sizeof(server_addr));

                if (bytes_sent < 0) {
                    fprintf(stderr,
                            "Worker %d: end packet sendto: %s\n",
                            ctx->worker_id,
                            strerror(errno));
                    ctx->error = 1;
                    break;
                }

                if ((size_t)bytes_sent != end_packet_len) {
                    fprintf(stderr,
                            "Worker %d: incomplete end packet send\n",
                            ctx->worker_id);
                    ctx->error = 1;
                    break;
                }

                usleep(1000);
            }
        }
    }

    close(sockfd);

    if (!ctx->error) {
        printf("Worker %d complete: %.2f MiB, %" PRIu64 " packets\n",
               ctx->worker_id,
               ctx->bytes_sent / (1024.0 * 1024.0),
               ctx->packets_sent);
        fflush(stdout);
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    const char *server_ip;
    const char *input_file;
    uint16_t base_port;
    int input_fd;
    struct stat input_stat;
    uint64_t file_size;
    pthread_t workers[THREAD_COUNT];
    sender_context_t contexts[THREAD_COUNT];
    struct timespec start_time;
    struct timespec end_time;
    uint64_t total_sent = 0;
    uint64_t total_packets = 0;
    int any_error = 0;

    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    server_ip = argv[1];
    input_file = argv[3];

    if (parse_base_port(argv[2], &base_port) != 0) {
        fprintf(stderr, "Invalid base port: %s\n", argv[2]);
        return EXIT_FAILURE;
    }

    input_fd = open(input_file, O_RDONLY);
    if (input_fd < 0) {
        fprintf(stderr,
                "open %s: %s\n",
                input_file,
                strerror(errno));
        return EXIT_FAILURE;
    }

    if (fstat(input_fd, &input_stat) < 0) {
        fprintf(stderr,
                "fstat %s: %s\n",
                input_file,
                strerror(errno));
        close(input_fd);
        return EXIT_FAILURE;
    }

    if (input_stat.st_size <= 0) {
        fprintf(stderr, "Input file must not be empty\n");
        close(input_fd);
        return EXIT_FAILURE;
    }

    file_size = (uint64_t)input_stat.st_size;

    printf("Four-thread UDP client\n");
    printf("Input file: %s\n", input_file);
    printf("File size: %" PRIu64 " bytes (%.2f MiB)\n",
           file_size,
           file_size / (1024.0 * 1024.0));
    printf("Server: %s\n", server_ip);
    printf("Server ports: %u through %u\n",
           base_port,
           (uint16_t)(base_port + THREAD_COUNT - 1));
    printf("Worker count: %d\n", THREAD_COUNT);
    printf("Payload size: %d bytes\n", PAYLOAD_SIZE);
    printf("Pacing delay per worker: %d microseconds\n",
           SEND_DELAY_US);
    fflush(stdout);

    for (int i = 0; i < THREAD_COUNT; i++) {
        memset(&contexts[i], 0, sizeof(contexts[i]));

        contexts[i].worker_id = i;
        contexts[i].cpu_id = i;
        contexts[i].input_fd = input_fd;
        contexts[i].start_offset =
            (file_size * (uint64_t)i) / THREAD_COUNT;
        contexts[i].end_offset =
            (file_size * (uint64_t)(i + 1)) / THREAD_COUNT;
        contexts[i].server_ip = server_ip;
        contexts[i].server_port = (uint16_t)(base_port + i);
    }

    if (clock_gettime(CLOCK_MONOTONIC, &start_time) != 0) {
        fprintf(stderr, "clock_gettime: %s\n", strerror(errno));
        close(input_fd);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < THREAD_COUNT; i++) {
        int result;

        result = pthread_create(&workers[i],
                                NULL,
                                sender_worker,
                                &contexts[i]);

        if (result != 0) {
            errno = result;
            fprintf(stderr,
                    "pthread_create worker %d: %s\n",
                    i,
                    strerror(errno));

            for (int j = 0; j < i; j++) {
                pthread_join(workers[j], NULL);
            }

            close(input_fd);
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < THREAD_COUNT; i++) {
        int result = pthread_join(workers[i], NULL);

        if (result != 0) {
            errno = result;
            fprintf(stderr,
                    "pthread_join worker %d: %s\n",
                    i,
                    strerror(errno));
            any_error = 1;
        }

        total_sent += contexts[i].bytes_sent;
        total_packets += contexts[i].packets_sent;

        if (contexts[i].error) {
            any_error = 1;
        }
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end_time) != 0) {
        fprintf(stderr, "clock_gettime: %s\n", strerror(errno));
        any_error = 1;
    }

    close(input_fd);

    {
        double elapsed = elapsed_seconds(&start_time, &end_time);
        double throughput_mbps = 0.0;

        if (elapsed > 0.0) {
            throughput_mbps =
                (total_sent * 8.0) / (elapsed * 1000000.0);
        }

        printf("\nClient transfer summary\n");
        printf("Bytes sent: %" PRIu64 "\n", total_sent);
        printf("Packets sent: %" PRIu64 "\n", total_packets);
        printf("Elapsed time: %.3f seconds\n", elapsed);
        printf("Payload throughput: %.2f Mbit/s\n", throughput_mbps);

        if (total_sent != file_size) {
            fprintf(stderr,
                    "ERROR: sent byte count does not match input file size\n");
            any_error = 1;
        }
    }

    if (any_error) {
        fprintf(stderr, "Client transfer failed\n");
        return EXIT_FAILURE;
    }

    printf("Client transfer completed successfully\n");
    return EXIT_SUCCESS;
}