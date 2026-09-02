#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define BUFFER_SIZE 1400

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{
    int sockfd, portno, filefd;
    ssize_t bytes_read, bytes_sent;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    char buffer[BUFFER_SIZE];

    if (argc < 4) {
        fprintf(stderr, "usage: %s hostname port file\n", argv[0]);
        exit(1);
    }

    portno = atoi(argv[2]);

    filefd = open(argv[3], O_RDONLY);
    if (filefd < 0)
        error("ERROR opening input file");

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

    while ((bytes_read = read(filefd, buffer, BUFFER_SIZE)) > 0) {
        bytes_sent = sendto(
            sockfd,
            buffer,
            bytes_read,
            0,
            (struct sockaddr *)&serv_addr,
            sizeof(serv_addr)
        );

        if (bytes_sent < 0)
            error("ERROR sending UDP packet");

        if (bytes_sent != bytes_read) {
            fprintf(stderr, "ERROR: incomplete UDP datagram sent\n");
            close(filefd);
            close(sockfd);
            exit(1);
        }
        usleep(25);
    }

    if (bytes_read < 0)
        error("ERROR reading input file");

    bytes_sent = sendto(
        sockfd,
        "EOF",
        3,
        0,
        (struct sockaddr *)&serv_addr,
        sizeof(serv_addr)
    );

    if (bytes_sent < 0)
        error("ERROR sending EOF packet");

    printf("UDP file send completed.\n");

    close(filefd);
    close(sockfd);

    return 0;
}
