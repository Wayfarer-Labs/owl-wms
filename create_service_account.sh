#!/bin/bash

# Create service account for SkyPilot Docker Registry Access
# This script creates a service account with Artifact Registry permissions

set -e  # Exit on any error

# Configuration
PROJECT_ID="openworld-main"
SERVICE_ACCOUNT_NAME="skypilot-docker"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="$HOME/gcp-key.json"

echo "Creating service account for SkyPilot Docker Registry access..."
echo "Project: $PROJECT_ID"
echo "Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "Key file: $KEY_FILE"
echo ""

# 1. Create the service account
echo "1. Creating service account..."
if gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL --project=$PROJECT_ID >/dev/null 2>&1; then
  echo "✓ Service account already exists: $SERVICE_ACCOUNT_EMAIL"
else
  gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="SkyPilot Docker Registry Access" \
    --project=$PROJECT_ID \
    --quiet
  echo "✓ Service account created: $SERVICE_ACCOUNT_EMAIL"
fi

# 2. Grant Artifact Registry reader permissions
echo "2. Granting Artifact Registry permissions..."
echo "   (waiting for service account to be ready...)"
sleep 5  # Wait for service account to be fully created

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
  --role="roles/artifactregistry.reader" \
  --quiet

echo "✓ Artifact Registry reader permissions granted"

# 3. Create and download the key file
echo "3. Creating service account key..."
gcloud iam service-accounts keys create $KEY_FILE \
  --iam-account=$SERVICE_ACCOUNT_EMAIL \
  --quiet

echo "✓ Service account key created: $KEY_FILE"

# 4. Show usage instructions
echo ""
echo "🎉 Service account setup complete!"
echo ""
echo "To use with SkyPilot:"
echo "  sky launch skypilot/config.yaml --env SKYPILOT_DOCKER_PASSWORD=\"\$(cat $KEY_FILE)\""
echo ""
echo "Or export the environment variable:"
echo "  export SKYPILOT_DOCKER_PASSWORD=\"\$(cat $KEY_FILE)\""
echo "  sky launch skypilot/config.yaml"
echo ""
echo "Key file location: $KEY_FILE"
echo "Service account: $SERVICE_ACCOUNT_EMAIL"