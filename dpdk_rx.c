#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>
#include <rte_pause.h>

#define MAGIC       UINT32_C(0x45453534)
#define UDP_PORT    5000
#define HDR_LEN     22
#define CHUNK       1400
#define FRAME_BYTES (224 * 224 * 3)
#define NFRAMES     16
#define NSEG        ((FRAME_BYTES + CHUNK - 1) / CHUNK)
#define BURST_SIZE  32

struct frame {
    uint8_t data[FRAME_BYTES];
    uint8_t seen[NSEG];
    unsigned int count;
};

/* Indexed by frame_id, regardless of packet arrival order. */
static struct frame frames[NFRAMES];

enum result {
    ACCEPTED,
    DUPLICATE,
    IGNORED,
    MALFORMED
};

/* Decode big-endian integers without requiring aligned pointers. */
static uint16_t read_be16(const uint8_t *p)
{
    return ((uint16_t)p[0] << 8) | p[1];
}

static uint32_t read_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           (uint32_t)p[3];
}

static enum result process_packet(const uint8_t *pkt, uint32_t size)
{
    /* Ethernet header: 14 bytes, EtherType at byte 12. */
    if (size < 14)
        return MALFORMED;

    if (read_be16(pkt + 12) != 0x0800)
        return IGNORED;  /* Only untagged IPv4 packets. */

    const uint8_t *ip = pkt + 14;
    uint32_t available = size - 14;

    if (available < 20 || (ip[0] >> 4) != 4)
        return MALFORMED;

    unsigned int ihl = (ip[0] & 0x0f) * 4;
    uint16_t ip_len = read_be16(ip + 2);

    if (ihl < 20 || ihl > available ||
        ip_len < ihl || ip_len > available)
        return MALFORMED;

    if (ip[9] != 17)
        return IGNORED;  /* IP protocol 17 = UDP. */

    /*
     * Reject fragmented IPv4 packets.
     * The generator's packets fit the MTU and need no fragmentation.
     * Mask includes the "more fragments" flag and fragment offset.
     */
    if (read_be16(ip + 6) & 0x3fff)
        return MALFORMED;

    if (ip_len - ihl < 8)
        return MALFORMED;

    const uint8_t *udp = ip + ihl;
    uint16_t udp_len = read_be16(udp + 4);

    if (udp_len < 8 || udp_len != ip_len - ihl)
        return MALFORMED;

    if (read_be16(udp + 2) != UDP_PORT)
        return IGNORED;

    const uint8_t *hdr = udp + 8;
    unsigned int payload_len = udp_len - 8;

    if (payload_len < HDR_LEN)
        return MALFORMED;

    if (read_be32(hdr) != MAGIC)
        return IGNORED;

    /* Python format: !IIHHIHI */
    uint32_t frame_id   = read_be32(hdr + 4);
    uint16_t seq        = read_be16(hdr + 8);
    uint16_t nseg       = read_be16(hdr + 10);
    uint32_t offset     = read_be32(hdr + 12);
    uint16_t length     = read_be16(hdr + 16);
    uint32_t frame_size = read_be32(hdr + 18);

    if (frame_id >= NFRAMES || frame_size != FRAME_BYTES ||
        nseg != NSEG || seq >= NSEG)
        return MALFORMED;

    /*
     * Validate against our generator's fixed chunk layout.
     * This also prevents overlaps and out-of-bounds copies.
     */
    uint32_t expected_offset = (uint32_t)seq * CHUNK;
    uint32_t expected_length = FRAME_BYTES - expected_offset;

    if (expected_length > CHUNK)
        expected_length = CHUNK;

    if (offset != expected_offset || length != expected_length ||
        payload_len != HDR_LEN + (unsigned int)length)
        return MALFORMED;

    struct frame *f = &frames[frame_id];

    if (f->seen[seq])
        return DUPLICATE;

    memcpy(f->data + offset, hdr + HDR_LEN, length);
    f->seen[seq] = 1;
    f->count++;

    return ACCEPTED;
}

static void check_dpdk(int rc, const char *operation)
{
    if (rc < 0)
        rte_exit(EXIT_FAILURE, "%s: %s\n",
                 operation, rte_strerror(-rc));
}

int main(int argc, char **argv)
{
    int used = rte_eal_init(argc, argv);

    if (used < 0)
        rte_exit(EXIT_FAILURE, "EAL initialization failed\n");

    argc -= used;
    argv += used;

    if (argc != 2)
        rte_exit(EXIT_FAILURE,
                 "Usage: dpdk_rx [EAL options] -- output.bin\n");

    const char *output_path = argv[1];

    uint16_t port;
    check_dpdk(rte_eth_dev_get_port_by_name("net_pcap0", &port),
               "Find net_pcap0");

    struct rte_mempool *pool = rte_pktmbuf_pool_create(
        "mbufs", 8191, 256, 0,
        RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    if (!pool)
        rte_exit(EXIT_FAILURE, "Create mbuf pool: %s\n",
                 rte_strerror(rte_errno));

    struct rte_eth_conf conf = {0};
    check_dpdk(rte_eth_dev_configure(port, 1, 1, &conf),
               "Configure port");

    uint16_t rx_desc = 1024;
    uint16_t tx_desc = 1024;

    check_dpdk(rte_eth_dev_adjust_nb_rx_tx_desc(
                   port, &rx_desc, &tx_desc),
               "Adjust queue descriptors");

    check_dpdk(rte_eth_rx_queue_setup(
                   port, 0, rx_desc, rte_socket_id(), NULL, pool),
               "Set up RX queue");

    check_dpdk(rte_eth_tx_queue_setup(
                   port, 0, tx_desc, rte_socket_id(), NULL),
               "Set up TX queue");

    check_dpdk(rte_eth_dev_start(port), "Start port");

    uint64_t received = 0;
    uint64_t counts[4] = {0};
    uint64_t last_packet = rte_get_timer_cycles();
    uint64_t idle_limit = 2 * rte_get_timer_hz();

    struct rte_mbuf *bufs[BURST_SIZE];

    printf("Receiving: expecting %d frames, %d chunks/frame\n",
           NFRAMES, NSEG);

    for (;;) {
        uint16_t n = rte_eth_rx_burst(port, 0, bufs, BURST_SIZE);

        if (n == 0) {
            if (rte_get_timer_cycles() - last_packet >= idle_limit)
                break;

            rte_pause();
            continue;
        }

        last_packet = rte_get_timer_cycles();

        for (uint16_t i = 0; i < n; i++) {
            struct rte_mbuf *m = bufs[i];
            uint32_t size = rte_pktmbuf_pkt_len(m);
            enum result result = MALFORMED;

            /*
             * Generator packets are at most 1464 bytes.
             * rte_pktmbuf_read also handles segmented mbufs.
             */
            uint8_t scratch[2048];

            if (size <= sizeof(scratch)) {
                const uint8_t *pkt =
                    rte_pktmbuf_read(m, 0, size, scratch);

                if (pkt)
                    result = process_packet(pkt, size);
            }

            received++;
            counts[result]++;
            rte_pktmbuf_free(m);
        }
    }

    struct rte_eth_stats stats = {0};
    int have_stats = rte_eth_stats_get(port, &stats) == 0;

    check_dpdk(rte_eth_dev_stop(port), "Stop port");
    check_dpdk(rte_eth_dev_close(port), "Close port");
    rte_mempool_free(pool);
    check_dpdk(rte_eal_cleanup(), "Clean up EAL");

    FILE *out = fopen(output_path, "wb");

    if (!out) {
        perror(output_path);
        return EXIT_FAILURE;
    }

    unsigned int complete = 0;

    /*
     * Write complete images in ascending frame_id order,
     * matching images.npy's original order.
     */
    for (unsigned int fid = 0; fid < NFRAMES; fid++) {
        if (frames[fid].count != NSEG) {
            fprintf(stderr, "Incomplete frame %u: %u/%d chunks\n",
                    fid, frames[fid].count, NSEG);
            continue;
        }

        if (fwrite(frames[fid].data, 1, FRAME_BYTES, out)
                != FRAME_BYTES) {
            perror("Write output");
            fclose(out);
            return EXIT_FAILURE;
        }

        complete++;
    }

    if (fclose(out) != 0) {
        perror("Close output");
        return EXIT_FAILURE;
    }

    printf("Packets received:      %" PRIu64 "\n", received);
    printf("Unique chunks copied:  %" PRIu64 "\n", counts[ACCEPTED]);
    printf("Duplicate chunks:      %" PRIu64 "\n", counts[DUPLICATE]);
    printf("Unrelated packets:     %" PRIu64 "\n", counts[IGNORED]);
    printf("Malformed packets:     %" PRIu64 "\n", counts[MALFORMED]);
    printf("Application discarded: %" PRIu64 "\n",
           counts[DUPLICATE] + counts[IGNORED] + counts[MALFORMED]);

    if (have_stats) {
        printf("Driver missed packets: %" PRIu64 "\n", stats.imissed);
        printf("Driver RX errors:      %" PRIu64 "\n", stats.ierrors);
        printf("Driver no-mbuf events: %" PRIu64 "\n", stats.rx_nombuf);
    }

    printf("Complete images:       %u/%d\n", complete, NFRAMES);
    printf("Output: %s (%zu bytes)\n",
           output_path, (size_t)complete * FRAME_BYTES);

    return complete == NFRAMES ? EXIT_SUCCESS : EXIT_FAILURE;
}
