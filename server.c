#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define BUFFER_SIZE 1400
#define FILE_SIZE_BYTES (1024ULL * 1024 * 1024)
#define REPORT_INTERVAL (10ULL * 1024 * 1024)

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{
    int sockfd, portno;
    ssize_t n;
    struct sockaddr_in serv_addr, client_addr;
    socklen_t client_len;
    char buffer[BUFFER_SIZE];
    struct timespec start_time, end_time;
    uint64_t bytes_received = 0;
    uint64_t last_reported = 0;
    int timer_started = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s port\n", argv[0]);
        exit(1);
    }

    portno = atoi(argv[1]);

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        error("ERROR opening UDP socket");

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR binding UDP socket");

    printf("UDP server listening on port %d\n", portno);
    printf("Waiting for file data from client...\n");

    client_len = sizeof(client_addr);

    while (1) {
        n = recvfrom(sockfd,
                     buffer,
                     sizeof(buffer),
                     0,
                     (struct sockaddr *)&client_addr,
                     &client_len);

        if (n < 0)
            error("ERROR receiving UDP data");

        if (n == 3 && memcmp(buffer, "EOF", 3) == 0)
            break;

        if (!timer_started) {
            clock_gettime(CLOCK_MONOTONIC, &start_time);
            timer_started = 1;
        }

        bytes_received += (uint64_t)n;

        if (bytes_received - last_reported >= REPORT_INTERVAL) {
            double mib = bytes_received / (1024.0 * 1024.0);
            double percent = (bytes_received * 100.0) / FILE_SIZE_BYTES;

            printf("Progress: %.2f MiB received (%.2f%%)\n", mib, percent);
            fflush(stdout);
            last_reported = bytes_received;
        }
    }

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
    } else {
        printf("No file data was received.\n");
    }

    close(sockfd);
    return 0;
}
