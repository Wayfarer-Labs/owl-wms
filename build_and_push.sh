#!/bin/bash

# Build, tag, and push Docker image script

set -e  # Exit on any error

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
else
    echo "Error: .env file not found. Copy .env.example to .env and configure it."
    exit 1
fi

# Check required environment variables
required_vars=("PROJECT_ID" "REGISTRY" "REPOSITORY" "IMAGE_NAME" "LOCAL_REGISTRY" "LOCAL_PROJECT" "LOCAL_REPOSITORY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set in .env"
        exit 1
    fi
done

# Set tag
TAG=${1:-${DEFAULT_TAG:-latest}}

# Constructed image names
LOCAL_TAG="${LOCAL_REGISTRY}/${LOCAL_PROJECT}/${LOCAL_REPOSITORY}:${TAG}"
REMOTE_TAG="${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image..."
docker build -t $LOCAL_TAG .

echo "Tagging image for remote repository..."
docker tag $LOCAL_TAG $REMOTE_TAG

echo "Pushing image to remote repository..."
docker push $REMOTE_TAG

echo "Getting image digest..."
DIGEST=$(docker inspect $REMOTE_TAG --format='{{index .RepoDigests 0}}' | cut -d'@' -f2)
if [ -z "$DIGEST" ]; then
    echo "Failed to get digest, falling back to tag"
    IMAGE_REF="$REMOTE_TAG"
else
    IMAGE_REF="${REMOTE_TAG%:*}@${DIGEST}"
fi

echo "Updating SkyPilot config with new image digest..."
sed -i "s|image_id: docker:.*|image_id: docker:$IMAGE_REF|" skypilot/config.yaml

echo "Build and push completed successfully!"
echo "Local tag: $LOCAL_TAG"
echo "Remote tag: $REMOTE_TAG"
echo "Image digest: $DIGEST"
echo "SkyPilot config updated to use: $IMAGE_REF"
echo ""
echo "To use this image:"
echo "Launch with sky launch skypilot/config.yaml"