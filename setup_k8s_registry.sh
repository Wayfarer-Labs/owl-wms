#!/bin/bash

# Setup Kubernetes to authenticate with GCP Artifact Registry
# This script creates the necessary secrets for pulling private images

set -e

# Configuration
REGISTRY_SERVER="us-central1-docker.pkg.dev"
SERVICE_ACCOUNT_EMAIL="skypilot-docker@openworld-main.iam.gserviceaccount.com"
SECRET_NAME="gcp-registry-secret"
NAMESPACE="default"  # Change if using different namespace
KEY_FILE="$HOME/gcp-key.json"

echo "Setting up Kubernetes authentication for GCP Artifact Registry..."
echo "Registry: $REGISTRY_SERVER"
echo "Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "Secret Name: $SECRET_NAME"
echo "Namespace: $NAMESPACE"
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Service account key file not found at $KEY_FILE"
    echo "Run ./create_service_account.sh first"
    exit 1
fi

# 1. Create Docker registry secret
echo "1. Creating Kubernetes secret for Docker registry..."
kubectl create secret docker-registry $SECRET_NAME \
  --docker-server=$REGISTRY_SERVER \
  --docker-username=_json_key \
  --docker-password="$(cat $KEY_FILE)" \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✓ Kubernetes secret created: $SECRET_NAME"

# 2. Update SkyPilot configuration
echo "2. Updating SkyPilot configuration..."
mkdir -p ~/.sky

# Create or update config.yaml with imagePullSecrets
cat > ~/.sky/config.yaml << EOF
kubernetes:
  pod_config:
    spec:
      imagePullSecrets:
        - name: $SECRET_NAME
EOF

echo "✓ SkyPilot configuration updated"

# 3. Verify the secret
echo "3. Verifying setup..."
kubectl get secret $SECRET_NAME -n $NAMESPACE >/dev/null 2>&1
echo "✓ Secret verified"

echo ""
echo "🎉 Kubernetes authentication setup complete!"
echo ""
echo "The Kubernetes cluster can now pull images from:"
echo "  $REGISTRY_SERVER"
echo ""
echo "You can now run:"
echo "  sky launch skypilot/config.yaml"
echo ""
echo "Secret name: $SECRET_NAME"
echo "Namespace: $NAMESPACE"