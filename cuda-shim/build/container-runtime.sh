#!/bin/sh
# Which Linux-container runtime runs nvcc on this Mac, printed as the command to call. Sourced or run by
# build/tinycc and build/fetch-cuda-headers.sh, so both agree on one answer rather than each detecting its own.
#
# Apple's `container` (macOS 26+, apple/container, `brew install container`) is preferred when it is present: it runs
# each container in its own lightweight VM on Virtualization.framework, needs no Docker Desktop, and takes the same
# `run --rm --network=none -v src:dst[:ro] image cmd` this project uses - checked flag by flag on macOS 27 / container
# 1.4.1, read-only mounts and all. Docker is used when it is not, and remains fully supported.
#
# CONTAINER_RUNTIME=<cmd> forces one either way; DOCKER=<cmd> is still honoured, since that is what this project's
# scripts took before Apple shipped a runtime at all.
#
# NOTE the one thing that does NOT carry across: Apple's CLI has no `cp` subcommand, so anything taking files out of
# an image does it with a bind mount and `run` (see fetch-cuda-headers.sh), which both runtimes do identically.
rt=${CONTAINER_RUNTIME:-${DOCKER:-}}
if [ -z "$rt" ]; then
  if command -v container > /dev/null 2>&1; then rt=container; else rt=docker; fi
fi
echo "$rt"
