# MedQCNN Roadmap

> Planning document for the next phases of development.
> Core system (quantum-classical pipeline, REST API, MCP, LangChain agent, basic frontend) is **complete**.

---

## Current State (Done)

- Hybrid quantum-classical CNN pipeline (ResNet-18 + PennyLane TorchLayer)
- REST API with `/health`, `/info`, `/predict` endpoints
- MCP server + LangChain agent integration
- Next.js frontend with single-image upload and diagnosis display
- Docker + Kafka deployment
- CI/CD (GitHub Actions: lint + test)
- 14 unit tests passing

---

## Phase 5: Interactive Frontend & Database

**Goal:** Make the frontend production-ready with persistent storage, batch upload, and training/benchmark dashboards.

### Sprint 5.1 — Database & Prediction History

- [ ] Add PostgreSQL database (via Docker Compose alongside Kafka)
- [ ] Define schema:
  - `predictions` table: id, image_filename, image_hash, prediction_label, confidence, probabilities (JSON), quantum_expectation_values (JSON), model_version, n_qubits, created_at
  - `training_runs` table: id, dataset, n_qubits, n_layers, epochs, learning_rate, batch_size, final_train_acc, final_val_acc, final_test_acc, auc_roc, f1, duration_seconds, checkpoint_path, created_at
  - `benchmarks` table: id, training_run_id, metric_name, metric_value, created_at
- [ ] Add ORM layer (SQLAlchemy or Tortoise ORM) to the backend
- [ ] Store every `/predict` result automatically in the database
- [ ] New API endpoints:
  - `GET /predictions` — list prediction history (paginated)
  - `GET /predictions/{id}` — single prediction detail
  - `GET /training-runs` — list training runs with metrics
  - `GET /benchmarks` — aggregated benchmark data

### Sprint 5.2 — Batch & Multi-Image Upload

- [ ] Increase upload limit from 10 MB to **250 MB** per request
- [ ] New API endpoint: `POST /predict/batch` — accepts multiple images in a single request
  - Input: array of base64-encoded images OR multipart form upload
  - Output: array of prediction results
  - Process images sequentially (or in parallel if RAM allows)
- [ ] Frontend: multi-file upload support
  - Drag-and-drop multiple files at once
  - File list with individual progress indicators
  - Upload limit indicator showing remaining capacity (X / 250 MB)
  - Select all / deselect individual files before submitting
- [ ] Frontend: batch results view
  - Summary card: X benign, Y malignant, average confidence
  - Table view with sortable columns (filename, label, confidence)
  - Click any row to expand full diagnosis detail (quantum values, probability bar)
- [ ] Image format support: PNG, JPG, BMP, TIFF, DICOM (.dcm)

### Sprint 5.3 — Training & Benchmark Dashboard

- [ ] Frontend: new `/training` page
  - List all training runs from database
  - Show per-run metrics: accuracy, AUC-ROC, F1, loss curves
  - Compare multiple runs side-by-side (e.g., 4-qubit vs 8-qubit)
- [ ] Frontend: new `/benchmarks` page
  - Parameter count comparison chart (quantum vs classical equivalent)
  - Memory usage by qubit count
  - Inference latency chart
  - Training convergence curves from stored runs
- [ ] Backend: auto-store training results
  - Modify `scripts/train.py` and `Trainer` to save run metadata to DB on completion
  - Modify `scripts/evaluate.py` to save evaluation metrics to DB
- [ ] Frontend: interactive charts (use Recharts or Chart.js)
  - Loss/accuracy curves
  - ROC curves
  - Confusion matrix heatmap
  - Quantum expectation value distribution across predictions

### Sprint 5.4 — Prediction History & UX Polish

- [ ] Frontend: new `/history` page
  - Paginated list of all past predictions
  - Filter by label (benign/malignant), date range, confidence threshold
  - Search by filename
  - Export as CSV
- [ ] Frontend: diagnosis detail page `/history/{id}`
  - Full prediction result with image preview
  - Quantum analysis breakdown
  - Link to the model version / training run that produced it
- [ ] Frontend: dark/light theme toggle
- [ ] Frontend: responsive mobile layout
- [ ] Loading skeletons for all data-fetching pages

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
| Migration | Alembic | Schema versioning |
| Charts | Recharts | Frontend data visualization |
| File Upload | multipart/form-data | Batch image upload (up to 250 MB) |

---

## Priority Order

1. **Sprint 5.1** — Database first (everything else depends on it)
2. **Sprint 5.2** — Batch upload (core user feature)
3. **Sprint 5.3** — Training dashboard (visibility into model performance)
4. **Sprint 5.4** — History & polish (completeness)
5. **Phase 6** — Security & production readiness
6. **Phase 7** — Quantum hardware & scale
