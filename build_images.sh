#!/bin/bash

TAG=0.0.5

PUSH_IMAGES=false
if [[ "$1" == "--push" ]]; then
    PUSH_IMAGES=true
fi

build_and_push() {
    platform=$1
    tag=$2
    version_tag=$3
    echo "Building image for $platform using base image $base_image..."
    docker build --platform "$platform" -t "$tag" -t "$version_tag" .
    if [ "$PUSH_IMAGES" = true ]; then
        echo "Pushing $tag to Docker Hub..."
        docker push "$tag"
        echo "Pushing $version_tag to Docker Hub..."
        docker push "$version_tag"
    fi
}

# Build CPU image for linux/arm64
build_and_push "linux/arm64" "owenpelliott/ingrain-server:latest-arm64" "owenpelliott/ingrain-server:$TAG-arm64"

# Build image for linux/amd64
build_and_push "linux/amd64" "owenpelliott/ingrain-server:latest-amd64" "owenpelliott/ingrain-server:$TAG-amd64"

# Check if docker manifest exists and create it if needed
echo "Creating multi-arch image..."
docker manifest rm owenpelliott/ingrain-server:latest 2>/dev/null
docker manifest rm owenpelliott/ingrain-server:$TAG 2>/dev/null

docker manifest create owenpelliott/ingrain-server:latest \
    --amend owenpelliott/ingrain-server:latest-arm64 \
    --amend owenpelliott/ingrain-server:latest-amd64

docker manifest create owenpelliott/ingrain-server:$TAG \
    --amend owenpelliott/ingrain-server:$TAG-arm64 \
    --amend owenpelliott/ingrain-server:$TAG-amd64

if [ "$PUSH_IMAGES" = true ]; then
    echo "Pushing multi-arch image manifest to Docker Hub..."
    docker manifest push owenpelliott/ingrain-server:latest
    docker manifest push owenpelliott/ingrain-server:$TAG
fi
