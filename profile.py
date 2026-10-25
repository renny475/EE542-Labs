"""
Two d7525 servers for EE 542 Lab 8 GPUNetIO experiments.

Instructions:
Wait until both nodes are ready, then use the SSH commands
shown on the experiment page.

Experiment addresses:
  node0: 10.10.1.1/24
  node1: 10.10.1.2/24

Each node has 100 GB of temporary storage mounted at /mydata.
Use this storage for Docker data.
Copy results off the nodes before terminating the experiment.
"""

import geni.portal as portal
import geni.rspec.pg as pg
import geni.rspec.emulab

request = portal.context.makeRequestRSpec()
lan = request.LAN("experiment-lan")

image = (
    "urn:publicid:IDN+emulab.net+image+"
    "emulab-ops//UBUNTU24-64-STD"
)

for index in range(2):
    node = request.RawPC(f"node{index}")
    node.hardware_type = "d7525"
    node.disk_image = image

    interface = node.addInterface("experiment")
    interface.addAddress(
        pg.IPv4Address(
            f"10.10.1.{index + 1}",
            "255.255.255.0",
        )
    )
    lan.addInterface(interface)

    storage = node.Blockstore(f"data{index}", "/mydata")
    storage.size = "100GB"

portal.context.printRequestRSpec()
