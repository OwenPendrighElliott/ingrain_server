#!/bin/bash
PUSH_IMAGES=false
if [[ "$1" == "--push" ]]; then
    PUSH_IMAGES=true
fi
build_and_push() {
    platform=$1
    tag=$2
    dockerfile=$3
    echo "Building $tag for $platform..."
    docker build --platform "$platform" -t "$tag" -f "$dockerfile" . --load
    if [ "$PUSH_IMAGES" = true ]; then
        echo "Pushing $tag to Docker Hub..."
        docker push "$tag"
    fi
}

# Build CPU base image for linux/arm64
build_and_push "linux/arm64" "owenpelliott/ingrain-base:arm64" "Dockerfile.base_arm64"
# Build GPU base image for linux/amd64
build_and_push "linux/amd64" "owenpelliott/ingrain-base:amd64" "Dockerfile.base_amd64"
