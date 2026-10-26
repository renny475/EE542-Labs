"""Part 7: two bare-metal d7525 nodes (A30 GPU + ConnectX-6) on one LAN.

node0 = 10.10.1.1, node1 = 10.10.1.2, netmask 255.255.255.0.
Each node mounts an extra file system at /mydata for Docker's data-root.

Put this file ALONE in a public GitHub repo and create the profile from it
in the DOCA-GPU project. Instantiate on the Wisconsin cluster.
"""

import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()
request = pc.makeRequestRSpec()

NODE_TYPE = "d7525"
IMAGE = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"
NETMASK = "255.255.255.0"

lan = request.LAN("lan0")
lan.best_effort = True          # don't fail the mapping on link bandwidth

for i, ip in enumerate(["10.10.1.1", "10.10.1.2"]):
    node = request.RawPC("node%d" % i)
    node.hardware_type = NODE_TYPE
    node.disk_image = IMAGE

    iface = node.addInterface("if%d" % i)
    iface.addAddress(pg.IPv4Address(ip, NETMASK))
    lan.addInterface(iface)

    # Extra file system for Docker (root disk is only 64 GB; DOCA image ~22 GB).
    bs = node.Blockstore("bs%d" % i, "/mydata")
    bs.size = "200GB"

pc.printRequestRSpec(request)
