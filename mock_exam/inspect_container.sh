#!/bin/bash
# Build the agent's Docker image from tasks/.../environment/Dockerfile
# and start an interactive shell inside it. Use this to confirm by hand
# what the frontier agent actually sees when it starts up — same image,
# same filesystem, same PATH, same $HOME.
#
# The container is ephemeral: nothing you do inside affects the host
# filesystem, and the container is deleted on exit (--rm).

set -euo pipefail

TASK="tasks/physical-sciences/chemistry/concentrated-electrolyte-transport"
IMAGE_TAG="diffec-mock-agent-view:latest"

[ -d "$TASK" ] || { echo "ERROR: task dir not found: $TASK"; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker not in PATH"; exit 2; }

echo "=== Building agent image from $TASK/environment/Dockerfile ==="
docker build -t "$IMAGE_TAG" "$TASK/environment"

echo
echo "=== Starting interactive shell inside the agent container ==="
echo "Try these to confirm isolation:"
echo "  ls /                                   # container root"
echo "  ls /root/data/cases/                   # the four case bundles"
echo "  find / -name CLAUDE.md 2>/dev/null     # confirm none of our repo state leaked"
echo "  find / -name solver.py -o -name reference_solver.py 2>/dev/null"
echo "  cat /root/data/cases/case_1/formalism.md | head -40"
echo "  exit                                    # tears down the container"
echo

docker run --rm -it "$IMAGE_TAG" /bin/bash
