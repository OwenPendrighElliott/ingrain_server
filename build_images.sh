#!/bin/bash
PUSH_IMAGES=false
if [[ "$1" == "--push" ]]; then
    PUSH_IMAGES=true
fi
build_and_push() {
    platform=$1
    base_image=$2
    tag=$3
    echo "Building image for $platform using base image $base_image..."
    docker build --platform "$platform" --build-arg BASE_IMAGE="$base_image" -t "$tag" .
    if [ "$PUSH_IMAGES" = true ]; then
        echo "Pushing $tag to Docker Hub..."
        docker push "$tag"
    fi
}
# Build CPU image for linux/amd64
build_and_push "linux/amd64" "owenpelliott/ingrain-cpu-base:amd64" "owenpelliott/ingrain-server:cpu-latest-amd64"
# Build CPU image for linux/arm64
build_and_push "linux/arm64" "owenpelliott/ingrain-cpu-base:arm64" "owenpelliott/ingrain-server:cpu-latest-arm64"
# Create multi-arch manifest for CPU image
echo "Creating multi-arch manifest for CPU image..."
docker manifest create owenpelliott/ingrain-server:cpu-latest \
  --amend owenpelliott/ingrain-server:cpu-latest-amd64 \
  --amend owenpelliott/ingrain-server:cpu-latest-arm64
# Push manifest if push flag is set
if [ "$PUSH_IMAGES" = true ]; then
    echo "Pushing multi-arch manifest for CPU image..."
    docker manifest push owenpelliott/ingrain-server:cpu-latest
fi
# Build GPU image for linux/amd64
build_and_push "linux/amd64" "owenpelliott/ingrain-gpu-base:amd64" "owenpelliott/ingrain-server:gpu-latest"
if [ "$PUSH_IMAGES" = true ]; then
    echo "Tagging and pushing GPU image as latest..."
    docker tag owenpelliott/ingrain-server:gpu-latest owenpelliott/ingrain-server:latest
    docker push owenpelliott/ingrain-server:gpu-latest
    docker push owenpelliott/ingrain-server:latest
fi
echo "Build process completed."
