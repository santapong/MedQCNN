# MedQCNN + KeepSave Integration Guide

This guide covers how to integrate MedQCNN with KeepSave for secure secret management, centralized MCP server hosting, OAuth authentication, and environment promotion.

## Table of Contents

1. [Overview](#overview)
2. [Secret Management](#secret-management)
3. [MCP Server Hub Registration](#mcp-server-hub-registration)
4. [OAuth 2.0 Authentication](#oauth-20-authentication)
5. [Environment Promotion Pipeline](#environment-promotion-pipeline)
6. [Python SDK Usage](#python-sdk-usage)
7. [Docker Deployment](#docker-deployment)
8. [CI/CD Integration](#cicd-integration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Integrate?

MedQCNN requires several sensitive configuration values to operate:

| Secret | Risk if Exposed |
|--------|----------------|
| `DATABASE_URL` | Database credentials leaked |
| `JWT_SECRET_KEY` | Token forgery, unauthorized API access |
| `OPENAI_API_KEY` | Unauthorized LLM usage, billing abuse |
| `KAFKA_BOOTSTRAP_SERVERS` | Message broker access |
| `MEDQCNN_API_KEY` | Unauthorized model inference |
| `CHECKPOINT_PATH` | Model location disclosure |

Storing these in `.env` files is risky — they can be accidentally committed to git, shared in logs, or accessed by unauthorized processes. KeepSave solves this by providing:

- **AES-256-GCM encryption** for all secrets at rest
- **Scoped API keys** so agents only access what they need
- **Audit trail** for every secret access
- **Environment promotion** to safely move configs between stages

### Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│    KeepSave API      │     │    MedQCNN API       │
│    (Port 8080)       │     │    (Port 8000)       │
│                      │     │                      │
│  Secret Vault        │◄───►│  Litestar Server     │
│  MCP Hub             │     │  Quantum Inference   │
│  OAuth Provider      │     │  MCP Server          │
│  API Key Manager     │     │  LangChain Agent     │
└──────────┬───────────┘     └──────────────────────┘
           │
           ▼
┌─────────────────────┐
│  AI Agents           │
│  (Claude, LangChain) │
│                      │
│  Call MedQCNN tools  │
│  via KeepSave        │
│  MCP Gateway         │
└─────────────────────┘
```

---

## Secret Management

### Prerequisites

1. KeepSave running at `http://localhost:8080`
2. A registered user account
3. The KeepSave Python SDK installed: `pip install keepsave`

### Step 1: Create a MedQCNN Project

```bash
# Register and login
curl -X POST http://localhost:8080/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@medqcnn.local", "password": "YourSecurePassword123!"}'

curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@medqcnn.local", "password": "YourSecurePassword123!"}'
# Save the returned JWT token

# Create project
curl -X POST http://localhost:8080/api/v1/projects \
  -H "Authorization: Bearer <jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "medqcnn", "description": "MedQCNN Hybrid Quantum-Classical CNN for Medical Diagnostics"}'
# Save the returned project ID
```

### Step 2: Store Secrets Per Environment

```bash
PROJECT_ID="<your-project-id>"
TOKEN="<your-jwt-token>"

# Alpha environment (development)
for secret in \
  "DATABASE_URL=sqlite:///medqcnn.db" \
  "JWT_SECRET_KEY=dev-secret-change-me" \
  "API_HOST=0.0.0.0" \
  "API_PORT=8000" \
  "N_QUBITS=4" \
  "MEDQCNN_AUTH_DISABLED=1"; do
  KEY="${secret%%=*}"
  VALUE="${secret#*=}"
  curl -X POST "http://localhost:8080/api/v1/projects/$PROJECT_ID/secrets" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"key\": \"$KEY\", \"value\": \"$VALUE\", \"environment\": \"alpha\"}"
done

# UAT environment (testing)
for secret in \
  "DATABASE_URL=postgresql://medqcnn:medqcnn@db:5432/medqcnn_uat" \
  "JWT_SECRET_KEY=$(openssl rand -base64 32)" \
  "API_HOST=0.0.0.0" \
  "API_PORT=8000" \
  "N_QUBITS=4" \
  "KAFKA_BOOTSTRAP_SERVERS=kafka:9092" \
  "CHECKPOINT_PATH=checkpoints/model_final.pt"; do
  KEY="${secret%%=*}"
  VALUE="${secret#*=}"
  curl -X POST "http://localhost:8080/api/v1/projects/$PROJECT_ID/secrets" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"key\": \"$KEY\", \"value\": \"$VALUE\", \"environment\": \"uat\"}"
done

# PROD environment (production)
for secret in \
  "DATABASE_URL=postgresql://medqcnn:STRONG_PASS@prod-db:5432/medqcnn" \
  "JWT_SECRET_KEY=$(openssl rand -base64 32)" \
  "API_HOST=0.0.0.0" \
  "API_PORT=8000" \
  "N_QUBITS=8" \
  "KAFKA_BOOTSTRAP_SERVERS=kafka-prod:9092" \
  "CHECKPOINT_PATH=checkpoints/model_prod_v1.pt"; do
  KEY="${secret%%=*}"
  VALUE="${secret#*=}"
  curl -X POST "http://localhost:8080/api/v1/projects/$PROJECT_ID/secrets" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"key\": \"$KEY\", \"value\": \"$VALUE\", \"environment\": \"prod\"}"
done
```

### Step 3: Create an API Key for MedQCNN

```bash
# Create a read-only API key scoped to the medqcnn project
curl -X POST http://localhost:8080/api/v1/api-keys \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "medqcnn-runtime", "project_id": "'$PROJECT_ID'", "scopes": ["read"]}'
# Save the returned API key (ks_xxxx)
```

### Step 4: Configure MedQCNN to Use KeepSave

Set only two environment variables on the MedQCNN host:

```bash
export KEEPSAVE_URL=http://localhost:8080
export KEEPSAVE_API_KEY=ks_xxxx
export MEDQCNN_ENV=alpha  # or uat, prod
```

---

## MCP Server Hub Registration

### Register MedQCNN in the Hub

```bash
curl -X POST http://localhost:8080/api/v1/mcp/servers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "medqcnn",
    "description": "Hybrid Quantum-Classical CNN for Medical Image Diagnostics. Tools: diagnose (image → prediction), model_info (architecture details), list_datasets (available MedMNIST benchmarks).",
    "github_url": "https://github.com/santapong/MedQCNN",
    "github_branch": "main",
    "entry_command": "uv run python scripts/mcp_server.py",
    "transport": "stdio",
    "is_public": true,
    "env_mappings": {
      "DATABASE_URL": "'$PROJECT_ID'/alpha/DATABASE_URL",
      "JWT_SECRET_KEY": "'$PROJECT_ID'/alpha/JWT_SECRET_KEY",
      "CHECKPOINT_PATH": "'$PROJECT_ID'/alpha/CHECKPOINT_PATH"
    }
  }'
```

### Install and Use via Gateway

```bash
# Install the MedQCNN MCP server
curl -X POST http://localhost:8080/api/v1/mcp/installations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mcp_server_id": "<medqcnn-server-id>"}'

# Call diagnose through the gateway
curl -X POST http://localhost:8080/api/v1/mcp/gateway \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "diagnose",
      "arguments": {"image_path": "/data/scans/patient001.png"}
    }
  }'

# Get model info
curl -X POST http://localhost:8080/api/v1/mcp/gateway \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "model_info", "arguments": {}}
  }'
```

### Claude Desktop / Claude Code Configuration

Get the auto-generated MCP config from KeepSave:

```bash
curl http://localhost:8080/api/v1/mcp/config \
  -H "Authorization: Bearer $TOKEN"
```

This returns a config block you can paste into your Claude client settings to access all installed MCP servers (including MedQCNN) through the KeepSave gateway.

---

## OAuth 2.0 Authentication

### Register MedQCNN as an OAuth Client

```bash
curl -X POST http://localhost:8080/api/v1/oauth/clients \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MedQCNN Diagnostic API",
    "redirect_uris": ["http://localhost:8000/auth/callback", "http://localhost:3000/auth/callback"],
    "scopes": ["read", "write"],
    "grant_types": ["authorization_code", "client_credentials"]
  }'
# Save client_id and client_secret
```

### Machine-to-Machine Auth (Agent → MedQCNN)

For AI agents that need to authenticate with MedQCNN via KeepSave:

```bash
# Get access token via client credentials
curl -X POST http://localhost:8080/api/v1/oauth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "ks_medqcnn_client",
    "client_secret": "ks_medqcnn_secret"
  }'

# Use the access token to call MedQCNN API
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "..."}'
```

### User-Facing Auth (Dashboard → MedQCNN)

For the MedQCNN web dashboard, use the authorization code flow with PKCE:

```
1. User clicks "Login with KeepSave" on MedQCNN dashboard
2. Redirect to: GET http://localhost:8080/api/v1/oauth/authorize
   ?response_type=code
   &client_id=ks_medqcnn_client
   &redirect_uri=http://localhost:3000/auth/callback
   &scope=read
   &code_challenge=<PKCE_challenge>
   &code_challenge_method=S256
3. User authenticates on KeepSave
4. Redirect back to MedQCNN with authorization code
5. Exchange code for tokens at POST /oauth/token
```

---

## Environment Promotion Pipeline

### Recommended Environment Strategy for MedQCNN

| Environment | Qubits | Database | Auth | Kafka | Use Case |
|-------------|--------|----------|------|-------|----------|
| **Alpha** | 4 (demo) | SQLite | Disabled | No | Local development |
| **UAT** | 4 | PostgreSQL | JWT enabled | Optional | Integration testing |
| **PROD** | 8 (full) | PostgreSQL | JWT + API keys | Yes | Production deployment |

### Promotion Workflow

```bash
# 1. Develop and test with Alpha secrets
keepsave pull --project medqcnn --env alpha --format env > .env
uv run python scripts/train.py --epochs 10 --n-qubits 4

# 2. Preview what changes when promoting to UAT
keepsave promote --project medqcnn --from alpha --to uat --dry-run

# 3. Promote to UAT (copies and re-encrypts secrets)
keepsave promote --project medqcnn --from alpha --to uat

# 4. Run integration tests with UAT config
MEDQCNN_ENV=uat uv run python -m pytest tests/

# 5. Promote to PROD (may require approval)
keepsave promote --project medqcnn --from uat --to prod --notes "v1.0 release"
```

---

## Python SDK Usage

### Basic Secret Loading

```python
"""Load MedQCNN secrets from KeepSave at startup."""
import os
from keepsave import KeepSaveClient

def load_secrets_from_keepsave():
    """Fetch and inject secrets from KeepSave into environment."""
    url = os.environ.get("KEEPSAVE_URL")
    api_key = os.environ.get("KEEPSAVE_API_KEY")

    if not url or not api_key:
        return  # Fall back to .env file

    client = KeepSaveClient(base_url=url, api_key=api_key)
    env = os.environ.get("MEDQCNN_ENV", "alpha")
    project_id = os.environ.get("KEEPSAVE_PROJECT_ID", "medqcnn")

    secrets = client.list_secrets(project_id, env)
    for secret in secrets:
        os.environ.setdefault(secret["key"], secret["value"])

# Call before importing MedQCNN modules
load_secrets_from_keepsave()
```

### Integration with MedQCNN Startup

```python
"""Example: scripts/serve_with_keepsave.py"""
import os

# Load secrets from KeepSave before anything else
from keepsave import KeepSaveClient

ks_url = os.environ.get("KEEPSAVE_URL", "http://localhost:8080")
ks_key = os.environ.get("KEEPSAVE_API_KEY")
ks_env = os.environ.get("MEDQCNN_ENV", "alpha")
ks_project = os.environ.get("KEEPSAVE_PROJECT_ID")

if ks_key and ks_project:
    client = KeepSaveClient(base_url=ks_url, api_key=ks_key)
    for secret in client.list_secrets(ks_project, ks_env):
        os.environ.setdefault(secret["key"], secret["value"])

# Now start MedQCNN normally
from medqcnn.api.server import create_app
import uvicorn

app = create_app()
uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", "8000")))
```

---

## Docker Deployment

### Combined Stack

```yaml
# docker-compose.keepsave.yml
version: "3.8"

services:
  # --- KeepSave Stack ---
  keepsave-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: keepsave
      POSTGRES_PASSWORD: keepsave_secure
      POSTGRES_DB: keepsave
    volumes:
      - keepsave-data:/var/lib/postgresql/data

  keepsave-api:
    build: /path/to/KeepSave
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://keepsave:keepsave_secure@keepsave-db:5432/keepsave
      MASTER_KEY: ${MASTER_KEY}
      JWT_SECRET: ${JWT_SECRET}
    depends_on:
      - keepsave-db

  # --- MedQCNN Stack ---
  medqcnn-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: medqcnn
      POSTGRES_PASSWORD: medqcnn_secure
      POSTGRES_DB: medqcnn
    volumes:
      - medqcnn-data:/var/lib/postgresql/data

  medqcnn-api:
    build: /path/to/MedQCNN
    ports:
      - "8000:8000"
    environment:
      KEEPSAVE_URL: http://keepsave-api:8080
      KEEPSAVE_API_KEY: ${KEEPSAVE_API_KEY}
      KEEPSAVE_PROJECT_ID: ${KEEPSAVE_MEDQCNN_PROJECT_ID}
      MEDQCNN_ENV: ${MEDQCNN_ENV:-alpha}
    depends_on:
      - keepsave-api
      - medqcnn-db
    volumes:
      - medqcnn-checkpoints:/app/checkpoints

volumes:
  keepsave-data:
  medqcnn-data:
  medqcnn-checkpoints:
```

---

## CI/CD Integration

### GitHub Actions: Pull Secrets for MedQCNN Tests

```yaml
# .github/workflows/test.yml
name: MedQCNN Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Pull secrets from KeepSave
        uses: santapong/keepsave-action@v1
        with:
          api-url: ${{ secrets.KEEPSAVE_URL }}
          api-key: ${{ secrets.KEEPSAVE_API_KEY }}
          project-id: ${{ secrets.KEEPSAVE_PROJECT_ID }}
          environment: alpha
          export-type: env

      - name: Run MedQCNN tests
        run: |
          uv sync --extra dev
          uv run python -m pytest tests/ -v
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `KeepSaveError: 401 Unauthorized` | API key expired or invalid | Regenerate API key in KeepSave dashboard |
| `KeepSaveError: 404 Not Found` | Wrong project ID or environment | Verify project ID and environment name |
| Secrets not loading | `KEEPSAVE_URL` or `KEEPSAVE_API_KEY` not set | Check environment variables are exported |
| MCP gateway timeout | MedQCNN MCP server not built | Check build status in KeepSave MCP Hub |
| OAuth token invalid | Wrong client credentials | Re-register OAuth client in KeepSave |
| Promotion blocked | PROD requires approval | Request approval from project admin |
