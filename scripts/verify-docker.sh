#!/usr/bin/env bash
# ──────────────────────────────────────────────
# MedQCNN — Docker Compose Verification Script
# Builds, starts, and validates all services.
#
# Usage:  ./scripts/verify-docker.sh
# ──────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

ERRORS=0

cleanup() {
    info "Cleaning up..."
    docker compose down -v --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── Step 1: Validate syntax ──────────────────
info "Validating docker-compose.yml syntax..."
if docker compose config --quiet 2>/dev/null; then
    pass "docker-compose.yml is valid"
else
    fail "docker-compose.yml has syntax errors"
    exit 1
fi

# ── Step 2: Build ─────────────────────────────
info "Building all services..."
if docker compose build; then
    pass "Docker build succeeded"
else
    fail "Docker build failed"
    exit 1
fi

# ── Step 3: Start services ────────────────────
info "Starting services..."
docker compose up -d

# ── Step 4: Wait for PostgreSQL ───────────────
info "Waiting for PostgreSQL..."
for i in $(seq 1 30); do
    if docker compose exec -T postgres pg_isready -U medqcnn > /dev/null 2>&1; then
        pass "PostgreSQL is ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail "PostgreSQL did not become ready"
        docker compose logs postgres
        ERRORS=$((ERRORS + 1))
    fi
    sleep 2
done

# ── Step 5: Wait for Kafka ────────────────────
info "Waiting for Kafka..."
for i in $(seq 1 30); do
    if docker compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
        pass "Kafka is ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail "Kafka did not become ready"
        docker compose logs kafka
        ERRORS=$((ERRORS + 1))
    fi
    sleep 2
done

# ── Step 6: Wait for API ──────────────────────
info "Waiting for MedQCNN API..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        pass "MedQCNN API is healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail "MedQCNN API did not pass health check"
        docker compose logs medqcnn-api
        ERRORS=$((ERRORS + 1))
    fi
    sleep 3
done

# ── Step 7: Verify API endpoints ──────────────
info "Testing API endpoints..."

if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    pass "GET /health returns 200"
else
    fail "GET /health failed"
    ERRORS=$((ERRORS + 1))
fi

if curl -sf http://localhost:8000/info > /dev/null 2>&1; then
    pass "GET /info returns 200"
else
    fail "GET /info failed"
    ERRORS=$((ERRORS + 1))
fi

# ── Step 8: Check containers ──────────────────
info "Checking container status..."
echo ""
docker compose ps
echo ""

if docker compose ps | grep -qE "Exit|Restarting"; then
    fail "Some containers are unhealthy or restarting"
    ERRORS=$((ERRORS + 1))
else
    pass "All containers are running"
fi

# ── Step 9: Show resource usage ───────────────
info "Resource usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" || true

# ── Summary ───────────────────────────────────
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}══════════════════════════════════════${NC}"
    echo -e "${GREEN}  All checks passed! Docker Compose   ${NC}"
    echo -e "${GREEN}  setup is working correctly.          ${NC}"
    echo -e "${GREEN}══════════════════════════════════════${NC}"
else
    echo -e "${RED}══════════════════════════════════════${NC}"
    echo -e "${RED}  $ERRORS check(s) failed.             ${NC}"
    echo -e "${RED}══════════════════════════════════════${NC}"
    exit 1
fi
