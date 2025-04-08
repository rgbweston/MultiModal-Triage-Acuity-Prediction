#!/bin/bash
# Install nvtop on the host
rpm -ivh ./nvtop-3.1.0-2.el9.x86_64.rpm

# Set up cdi for nvidia
mkdir /etc/cdi
nvidia-ctk cdi generate > /etc/cdi/nvidia.yaml

systemctl restart podman # unsure whether this is necessary
