# MedQCNN Roadmap

> Planning document for the next phases of development.
> Core system (quantum-classical pipeline, REST API, MCP, LangChain agent, frontend) is **complete** through Phase 5.

---

## Current State (Done)

- Hybrid quantum-classical CNN pipeline (ResNet-18 + PennyLane TorchLayer)
- REST API with `/health`, `/info`, `/predict`, `/predict/batch` endpoints
- Database layer (SQLAlchemy + PostgreSQL/SQLite) with auto-storage
- Prediction history, training runs, and benchmarks persistence
- New API endpoints: `/predictions`, `/training-runs`, `/benchmarks`
- MCP server + LangChain agent integration
- Next.js frontend with:
  - Single-image and batch upload with drag-and-drop
  - Prediction history page with filtering, pagination, CSV export
  - Training dashboard with interactive Recharts loss/accuracy curves
  - Benchmarks page with parameter count, memory, and latency charts
  - History detail page with quantum analysis breakdown
  - Dark/light theme toggle
  - Loading skeletons for all data-fetching pages
- Docker + Kafka + PostgreSQL deployment
- CI/CD (GitHub Actions: lint + test)
- 14 unit tests passing

---

## Phase 5: Interactive Frontend & Database ✅

**Goal:** Make the frontend production-ready with persistent storage, batch upload, and training/benchmark dashboards.

### Sprint 5.1 — Database & Prediction History ✅

- [x] Add PostgreSQL database (via Docker Compose alongside Kafka)
- [x] Define schema: `predictions`, `training_runs`, `benchmarks` tables
- [x] Add ORM layer (SQLAlchemy 2.x) to the backend
- [x] Store every `/predict` result automatically in the database
- [x] New API endpoints:
  - `GET /predictions` — list prediction history (paginated)
  - `GET /predictions/{id}` — single prediction detail
  - `GET /training-runs` — list training runs with metrics
  - `GET /benchmarks` — aggregated benchmark data

### Sprint 5.2 — Batch & Multi-Image Upload ✅

- [x] Increase upload limit from 10 MB to **250 MB** per request
- [x] New API endpoint: `POST /predict/batch` — accepts multiple images in a single request
- [x] Frontend: multi-file upload support with drag-and-drop
- [x] Frontend: batch results view with summary card and expandable table rows
- [x] Image format support: PNG, JPG, BMP, TIFF

### Sprint 5.3 — Training & Benchmark Dashboard ✅

- [x] Frontend: new `/training` page with training runs table and interactive charts
- [x] Frontend: new `/benchmarks` page with parameter count, memory, latency comparisons
- [x] Backend: auto-store training results in database on completion
- [x] Backend: auto-store evaluation results in database
- [x] Frontend: interactive Recharts charts (loss curves, accuracy curves)

### Sprint 5.4 — Prediction History & UX Polish ✅

- [x] Frontend: new `/history` page with paginated prediction list
- [x] Filters (label, date range, confidence), search by filename, CSV export
- [x] Detail page `/history/{id}` with quantum analysis breakdown
- [x] Dark/light theme toggle
- [x] Loading skeletons for all data-fetching pages

---

## Phase 6: Production Hardening

- [ ] Authentication (JWT or API key) on all endpoints
- [ ] Rate limiting on `/predict` and `/predict/batch`
- [ ] Full training benchmark: 50+ epochs on BreastMNIST (8 qubits) with stored results
- [ ] Model versioning: support multiple checkpoints, default to latest
- [ ] DICOM metadata parsing (patient anonymization, study info)
- [ ] Input sanitization and OWASP security audit
- [ ] Health check dashboard with uptime monitoring

## Phase 7: Quantum Hardware & Scale

- [ ] Test on Qiskit Aer simulator with noise models
- [ ] Deploy to IBM Quantum hardware via `pennylane-qiskit` backend
- [ ] Multi-class classification (PathMNIST 9-class, OrganAMNIST 11-class)
- [ ] 3D MRI volume processing (full NIfTI support)
- [ ] Raspberry Pi 5 cluster deployment validation
- [ ] RAG integration via ChromaDB for clinical knowledge retrieval

---

## Tech Stack Additions (Phase 5)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Database | PostgreSQL 16 | Prediction history, training runs, benchmarks |
| ORM | SQLAlchemy 2.x | Database models and queries |
| Charts | Recharts | Frontend data visualization |
| File Upload | multipart/form-data | Batch image upload (up to 250 MB) |

---

## Priority Order

1. ~~**Sprint 5.1** — Database first~~ ✅
2. ~~**Sprint 5.2** — Batch upload~~ ✅
3. ~~**Sprint 5.3** — Training dashboard~~ ✅
4. ~~**Sprint 5.4** — History & polish~~ ✅
5. **Phase 6** — Security & production readiness
6. **Phase 7** — Quantum hardware & scale
