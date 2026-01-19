#!/usr/bin/env bash
#
# Langfuse Setup Script for Tool Orchestrator
#
# This script sets up Vault policy and stores Langfuse API keys
# for the tool-orchestrator service.
#
# Usage:
#   ./scripts/langfuse-setup.sh
#
# Prerequisites:
#   - vault CLI installed and configured
#   - VAULT_ADDR environment variable set
#   - Authenticated to Vault with appropriate permissions
#

set -euo pipefail

SERVICE_NAME="tool-orchestrator"
VAULT_PATH="secret/${SERVICE_NAME}/langfuse"
POLICY_NAME="${SERVICE_NAME}-policy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    if ! command -v vault &> /dev/null; then
        error "vault CLI is not installed. Please install it first."
    fi

    if [[ -z "${VAULT_ADDR:-}" ]]; then
        error "VAULT_ADDR environment variable is not set."
    fi

    # Check if authenticated
    if ! vault token lookup &> /dev/null; then
        error "Not authenticated to Vault. Please run 'vault login' first."
    fi

    info "Prerequisites check passed"
}

# Create and register Vault policy
setup_vault_policy() {
    info "Setting up Vault policy: ${POLICY_NAME}"

    # Create policy inline
    vault policy write "${POLICY_NAME}" - <<EOF
# Policy for ${SERVICE_NAME} service
# Allows read access to Langfuse secrets

path "secret/data/${SERVICE_NAME}/*" {
  capabilities = ["read"]
}
EOF

    info "Vault policy '${POLICY_NAME}' created successfully"
}

# Prompt for and store Langfuse keys
store_langfuse_keys() {
    echo ""
    echo "=========================================="
    echo "  Langfuse API Key Configuration"
    echo "=========================================="
    echo ""
    echo "Enter your Langfuse API keys."
    echo "You can find these in your Langfuse dashboard under Settings > API Keys"
    echo ""

    # Prompt for public key
    read -rp "Langfuse Public Key (pk-lf-...): " public_key

    if [[ -z "${public_key}" ]]; then
        error "Public key cannot be empty"
    fi

    if [[ ! "${public_key}" =~ ^pk- ]]; then
        warn "Public key doesn't start with 'pk-'. Are you sure this is correct?"
        read -rp "Continue anyway? (y/N): " confirm
        if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
            error "Aborted by user"
        fi
    fi

    # Prompt for secret key (hidden input)
    read -rsp "Langfuse Secret Key (sk-lf-...): " secret_key
    echo ""

    if [[ -z "${secret_key}" ]]; then
        error "Secret key cannot be empty"
    fi

    if [[ ! "${secret_key}" =~ ^sk- ]]; then
        warn "Secret key doesn't start with 'sk-'. Are you sure this is correct?"
        read -rp "Continue anyway? (y/N): " confirm
        if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
            error "Aborted by user"
        fi
    fi

    # Store in Vault
    info "Storing Langfuse keys in Vault at ${VAULT_PATH}"

    vault kv put "${VAULT_PATH}" \
        public_key="${public_key}" \
        secret_key="${secret_key}"

    info "Langfuse keys stored successfully"
}

# Verify the stored keys
verify_keys() {
    info "Verifying stored keys..."

    if vault kv get -field=public_key "${VAULT_PATH}" &> /dev/null; then
        info "Verification successful - keys are accessible"
    else
        error "Verification failed - could not read keys from Vault"
    fi
}

# Main
main() {
    echo ""
    echo "=========================================="
    echo "  Tool Orchestrator - Langfuse Setup"
    echo "=========================================="
    echo ""

    check_prerequisites
    setup_vault_policy
    store_langfuse_keys
    verify_keys

    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Langfuse tracing is now configured."
    echo ""
    echo "Next steps:"
    echo "  1. Deploy the service: make deploy"
    echo "  2. Check logs for 'LANGFUSE OBSERVABILITY: ENABLED'"
    echo ""
    echo "To update keys later, run this script again or:"
    echo "  vault kv put ${VAULT_PATH} public_key=<key> secret_key=<key>"
    echo ""
}

main "$@"
