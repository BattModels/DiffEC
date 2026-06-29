#!/bin/bash
# Build the agent + verifier Docker images and push them to GHCR so
# Singularity on an HPC cluster (e.g. Artemis) can pull them by name.
#
# Run this ONCE on a Docker-capable host (laptop / Codespaces).
# After it succeeds, edit task.toml to add the printed
# `docker_image = "ghcr.io/…"` lines, commit, and run the pilot on
# Artemis via scripts/pilot/run_pilot_singularity.sh.
#
# Usage:
#     GHCR_USER=<your-github-handle> bash scripts/build_and_push.sh
#     # optional: TAG=v2 to override the default tag
#
# Authentication:
#   - If `gh` CLI is installed and authenticated, the script uses
#     `gh auth token` automatically.
#   - Otherwise, set GHCR_TOKEN to a GitHub personal-access token
#     with the `write:packages` scope:
#         export GHCR_TOKEN=ghp_xxx...
#         GHCR_USER=… bash scripts/build_and_push.sh

set -euo pipefail

GHCR_USER="${GHCR_USER:?Set GHCR_USER to your GitHub handle (lowercase)}"
TAG="${TAG:-v1}"
TASK="tasks/physical-sciences/chemistry/concentrated-electrolyte-mass-transport"

# GHCR requires lowercase repo names.
GHCR_USER_LC="$(echo "$GHCR_USER" | tr '[:upper:]' '[:lower:]')"
ENV_IMAGE="ghcr.io/${GHCR_USER_LC}/diffec-env:${TAG}"
TESTS_IMAGE="ghcr.io/${GHCR_USER_LC}/diffec-tests:${TAG}"

command -v docker >/dev/null || { echo "ERROR: docker required (not on this host)."; exit 2; }
[ -d "$TASK/environment" ] || { echo "ERROR: $TASK/environment not found."; exit 2; }
[ -d "$TASK/tests" ]       || { echo "ERROR: $TASK/tests not found."; exit 2; }

echo "=== Logging in to ghcr.io as $GHCR_USER_LC ==="
if [ -n "${GHCR_TOKEN:-}" ]; then
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER_LC" --password-stdin
elif command -v gh >/dev/null; then
    gh auth token | docker login ghcr.io -u "$GHCR_USER_LC" --password-stdin
else
    echo "ERROR: need either GHCR_TOKEN env var or 'gh auth login'."
    exit 2
fi

echo
echo "=== Building agent env image: $ENV_IMAGE ==="
docker build --platform linux/amd64 -t "$ENV_IMAGE" "$TASK/environment"

echo
echo "=== Building verifier image: $TESTS_IMAGE ==="
docker build --platform linux/amd64 -t "$TESTS_IMAGE" "$TASK/tests"

echo
echo "=== Pushing $ENV_IMAGE ==="
docker push "$ENV_IMAGE"

echo
echo "=== Pushing $TESTS_IMAGE ==="
docker push "$TESTS_IMAGE"

cat <<EOF

================================================================
Build and push complete.

Next steps:
1) Make both images PUBLIC so Singularity on Artemis can pull
   them anonymously (no token needed inside the cluster):

   gh api --method PATCH \\
       /user/packages/container/diffec-env/visibility \\
       -f visibility=public
   gh api --method PATCH \\
       /user/packages/container/diffec-tests/visibility \\
       -f visibility=public

   (Or open https://github.com/$GHCR_USER_LC?tab=packages
   in a browser, click each package, Package settings → Change
   visibility → Public.)

2) Add these two lines to task.toml:

   [environment]
   docker_image = "$ENV_IMAGE"
   # …keep the existing cpus / memory_mb / etc lines below…

   [verifier.environment]
   docker_image = "$TESTS_IMAGE"

   Note: the Dockerfile-based fields are still respected by the
   Docker env mode (Harbor's upstream-default), so keep them too —
   the two modes coexist. \`docker_image\` is just an additional
   hint that Singularity uses.

3) Commit the task.toml change and the new images are ready
   for the pilot:

   bash scripts/pilot/run_pilot_singularity.sh

================================================================
EOF
