# Enterprise NeuroMove Platform - Implementation Plan

## Executive Summary

Transform the current research-grade MI-BCI codebase into a **production-grade, enterprise-ready platform** comparable to Google/Amazon/Microsoft client-facing products. This plan covers architecture, security, scalability, compliance, and delivery.

---

## 1. Current State Assessment

### What Exists (Strengths)
- ✅ 6 neural architectures (MiniRocket, EEGNet, CNN-LSTM, CNN+Transformer, Shallow ConvNet, Advanced EEG)
- ✅ 6 datasets (PhysioNet, BCI IV 2a, High-Gamma, Kaya, NEMAR, WAY-EEG-GAL)
- ✅ Streamlit dashboard with professional styling
- ✅ FastAPI backend skeleton
- ✅ React + Vite frontend scaffold
- ✅ Cross-subject validation (LOSO) methodology
- ✅ Few-shot subject calibration

### Critical Gaps (Production Blockers)
| Gap | Impact | Priority |
|-----|--------|----------|
| No authentication/authorization | Security failure | P0 |
| No database persistence | No audit trail, no multi-user | P0 |
| No async job queue | Training blocks API | P0 |
| No observability | Cannot debug production | P0 |
| No CI/CD | Manual deployments | P1 |
| No automated testing | Regression risk | P1 |
| No compliance framework | Cannot sell to enterprise | P1 |
| No API gateway | No rate limiting, auth proxy | P1 |
| No secrets management | Credentials in code | P0 |

---

## 2. Target Architecture (Enterprise Standard)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXTERNAL CLIENTS                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │   React     │  │   Mobile    │  │   CLI/SDK   │  │  3rd Party API     │  │
│  │   SPA       │  │   Apps      │  │  (Python/JS)│  │  Consumers         │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬───────────┘  │
└─────────┼────────────────┼────────────────┼─────────────────┼────────────────┘
          │                │                │                 │
          ▼                ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API GATEWAY (Kong / AWS API GW)                     │
│  • TLS Termination  • Rate Limiting  • Auth Proxy  • Request Validation     │
│  • Request Routing  • Circuit Breaker • Request/Response Transform         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI APPLICATION CLUSTER                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│  │   Auth     │ │  Projects  │ │  Training  │ │ Inference  │ │  Datasets  │ │
│  │  Service   │ │  Service   │ │  Service   │ │  Service   │ │  Service   │ │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│  │  Models    │ │  Audits    │ │  Health    │ │  Config    │ │  Webhooks  │ │
│  │  Registry  │ │  Service   │ │  Checks    │ │  Service   │ │  Service   │ │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  PostgreSQL   │      │      Redis       │      │  RabbitMQ /     │
│  (Primary DB) │      │  (Cache/Sessions)│      │  Kafka (Async)  │
└───────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  S3/GCS       │      │  MLflow /       │      │  Prometheus /   │
│  (Artifacts)  │      │  Model Registry │      │  Grafana / Loki │
└───────────────┘      └─────────────────┘      └─────────────────┘
```

---

## 3. Implementation Phases

### Phase 0: Foundation (Week 1-2) ✅ IN PROGRESS
- [x] Fresh git repo with clean history
- [x] Professional `.gitignore`
- [ ] **Project structure standardization**
- [ ] **Development environment** (Docker Compose, dev containers)
- [ ] **Pre-commit hooks** (ruff, black, mypy, prettier)
- [ ] **Conventional commits** + semantic release

### Phase 1: Core Backend (Week 2-4)
| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | **FastAPI App Factory** with lifespan, DI container | `app/main.py`, `app/core/` |
| 1.2 | **PostgreSQL + AsyncPG + SQLAlchemy 2.0** + Alembic | Migrations, models |
| 1.3 | **Redis** for caching, sessions, rate limiting | `app/core/cache.py` |
| 1.4 | **Auth Service** - OAuth2/OIDC (Keycloak), JWT, RBAC | `app/api/v1/auth.py` |
| 1.5 | **API Gateway Integration** (Kong) | Kong config, plugins |
| 1.5 | **Pydantic v2 schemas** for all APIs | `app/schemas/` |
| 1.6 | **Exception handling, logging, middleware** | Structured logging, correlation IDs |
| 1.7 | **OpenAPI/Swagger** auto-generation | `/docs`, `/redoc` |

### Phase 2: Core Services (Week 4-6)
| Service | Responsibilities | Key Endpoints |
|---------|------------------|---------------|
| **Auth** | Login, register, MFA, SSO, token refresh | `POST /auth/login`, `POST /auth/refresh`, `GET /auth/me` |
| **Projects** | CRUD, members, permissions | `CRUD /projects`, `POST /projects/{id}/members` |
| **Datasets** | Upload, validation, versioning, preprocessing | `POST /datasets`, `GET /datasets/{id}/preview` |
| **Training** | Async job queue, progress tracking, artifacts | `POST /training/jobs`, `WS /training/jobs/{id}/progress` |
| **Inference** | Sync/async prediction, batch, streaming | `POST /inference`, `POST /inference/batch` |
| **Models** | Registry, versioning, promotion, deployment | `CRUD /models`, `POST /models/{id}/promote` |
| **Datasets** | Versioning, preprocessing pipelines | `POST /datasets`, `GET /datasets/{id}/export` |
| **Audits** | Immutable audit log, compliance export | `GET /audits`, `GET /audits/export` |

### Phase 3: ML Pipeline & Model Registry (Week 6-8)
| Component | Technology | Features |
|-----------|------------|----------|
| **Training Orchestrator** | Celery + RabbitMQ | Distributed training, GPU scheduling |
| **Model Registry** | MLflow + S3 | Versioning, lineage, promotion gates |
| **Experiment Tracking** | MLflow | Params, metrics, artifacts, reproducibility |
| **Model Serving** | Triton / TorchServe | Batch, streaming, A/B testing |
| **Feature Store** | Feast | Feature versioning, point-in-time correctness |

### Phase 4: Frontend (Week 8-10)
| Component | Technology | Standards |
|-----------|------------|-----------|
| **Design System** | Tailwind + Radix UI | Consistent, accessible (WCAG 2.1 AA) |
| **State Management** | TanStack Query + Zustand | Server state + client state |
| **API Layer** | TanStack Query + OpenAPI client | Type-safe, auto-generated |
| **Testing** | Vitest + Playwright | Unit, integration, E2E |
| **Storybook** | Component documentation | Design system docs |

### Phase 5: Security & Compliance (Week 10-12)
| Control | Implementation | Verification |
|---------|----------------|--------------|
| **Authentication** | OAuth2/OIDC + PKCE, MFA, passkeys | Auth0/Keycloak audit |
| **Authorization** | RBAC + ABAC, resource-level permissions | Policy unit tests |
| **Data Protection** | Encryption at rest (AES-256), in transit (TLS 1.3) | Key rotation policy |
| **Secrets Management** | HashiCorp Vault / AWS Secrets Manager | Rotation, audit |
| **Audit Logging** | Immutable append-only, tamper-evident | Compliance export |
| **Rate Limiting** | Token bucket, per-tenant, per-endpoint | Load test verification |
| **Vulnerability Scanning** | Trivy, Snyk, Dependabot | CI gate |
| **Penetration Testing** | Annual third-party | Report + remediation |

### Phase 6: Observability & Operations (Week 12-14)
| Pillar | Stack | Targets |
|--------|-------|---------|
| **Metrics** | Prometheus + Grafana | <1% error rate, <200ms p99 |
| **Logs** | Loki + Grafana | Structured JSON, correlation IDs |
| **Traces** | Tempo / Jaeger | Distributed tracing |
| **Alerting** | Alertmanager + PagerDuty | <5min MTTA |
| **SLO/SLI** | Error budget, burn rate alerts | 99.9% availability |
| **Runbooks** | GitHub Wiki + Grafana dashboards | <15min MTTR |

### Phase 7: CI/CD & Deployment (Week 14-16)
| Pipeline | Tool | Stages |
|----------|------|--------|
| **CI** | GitHub Actions | Lint → Typecheck → Unit → Integration → Security Scan → Build |
| **CD** | ArgoCD / Flux | Staging → Canary → Production |
| **Infrastructure** | Terraform | AWS/GCP modules |
| **Secrets** | SOPS + Age / Vault | GitOps secrets |
| **Rollback** | ArgoCD rollback | <2min |

### Phase 8: Compliance & Documentation (Week 16-18)
| Artifact | Audience | Format |
|----------|----------|--------|
| **Architecture Decision Records** | Engineering | Markdown (adr/) |
| **API Reference** | Developers | OpenAPI 3.1 + Redoc |
| **User Guide** | Clinicians/Researchers | MkDocs |
| **Admin Guide** | DevOps | Runbooks |
| **Security Whitepaper** | Enterprise Buyers | PDF |
| **HIPAA/GDPR Assessment** | Compliance | Assessment Report |
| **SOC2 Type II Readiness** | Auditors | Evidence Package |

---

## 4. Security Hardening Checklist

### Application Security
- [ ] **Input Validation** - Pydantic v2 on all inputs, strict schemas
- [ ] **Output Encoding** - Auto-escaping in templates, CSP headers
- [ ] **CSRF Protection** - Double-submit cookie + SameSite=Strict
- [ ] **CORS Policy** - Strict allowlist, no wildcards
- [ ] **Security Headers** - HSTS, X-Frame-Options, X-Content-Type-Options, Permissions-Policy
- [ ] **Rate Limiting** - Per-user, per-endpoint, token bucket
- [ ] **Request Size Limits** - Body, header, file upload limits
- [ ] **File Upload Security** - Type validation, virus scan, sandboxed processing

### Data Protection
- [ ] **Encryption at Rest** - AES-256-GCM for PII, keys in Vault
- [ ] **Encryption in Transit** - TLS 1.3 only, cert pinning for mobile
- [ ] **Key Rotation** - Automated 90-day rotation, HSM-backed
- [ ] **Data Minimization** - Collect only required fields
- [ ] **Right to Erasure** - GDPR Article 17 implementation
- [ ] **Data Retention** - Automated purge policies

### Infrastructure Security
- [ ] **Network Segmentation** - Private subnets, NAT gateways
- [ ] **WAF** - AWS WAF / Cloudflare rules
- [ ] **DDoS Protection** - Shield Advanced / Cloudflare
- [ ] **Vulnerability Scanning** - Trivy (container), Snyk (deps), Semgrep (code)
- [ ] **Container Hardening** - Distroless, non-root, read-only FS
- [ ] **Runtime Security** - Falco / Tetragon

---

## 4. Testing Strategy

| Layer | Tool | Coverage Target | Gates |
|-------|------|-----------------|-------|
| **Unit** | pytest + pytest-asyncio | ≥90% | Required |
| **Contract** | Pact / Schemathesis | All endpoints | Required |
| **Integration** | Testcontainers (Postgres, Redis, RabbitMQ) | Critical paths | Required |
| **E2E** | Playwright | Critical user journeys | Required |
| **Performance** | k6 / Locust | p99 < 200ms | Required |
| **Chaos** | Chaos Mesh | Dependency failures | Optional |
| **Security** | Bandit, Semgrep, Trivy | Zero critical | Required |

---

## 5. Development Standards

### Code Quality
```yaml
# pyproject.toml standards
[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "B", "C4", "UP", "PDF", "PGH", "PLC", "PLE", "PLR", "PLW", "TRY", "DTZ", "G", "LOG", "ARG", "PTH", "ERA", "COM", "SIM", "TCH", "TID", "EXE", "NPY", "PD", "PIE", "PT", "Q", "RET", "RSE", "SLF", "SLOT", "TRY", "UNUSED", "YTT"]
ignore = ["E501", "B008"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true

[tool.black]
line-length = 100
target-version = ['py311']
```

### Git Workflow
```
main (protected) ← PR ← feature/* (or fix/*, chore/*, docs/*)
         ↑
    Required: CI pass + 2 approvals + linear history
```

### Commit Convention
```
feat(api): add dataset versioning endpoint
fix(auth): handle expired refresh token gracefully
docs: update API reference for inference endpoint
chore(deps): upgrade fastapi to 0.115
refactor(services): extract training orchestrator
test(inference): add batch prediction contract test
```

---

## 5. Infrastructure as Code

### Terraform Structure
```
infra/
├── modules/
│   ├── vpc/
│   ├── rds/
│   ├── elasticache/
│   ├── eks/
│   ├── kafka/
│   ├── s3/
│   ├── secrets/
│   └── monitoring/
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
└── root.tf
```

### Kubernetes (EKS/GKE)
```yaml
# Base manifests with Kustomize
base/
├── deployment.yaml
├── service.yaml
├── ingress.yaml
├── hpa.yaml
├── pdb.yaml
├── networkpolicy.yaml
└── servicemonitor.yaml

overlays/
├── dev/
├── staging/
└── prod/
```

---

## 6. Cost Estimation (Monthly, USD)

| Component | Dev | Staging | Prod |
|-----------|-----|---------|------|
| EKS (3x m6i.large) | $150 | $450 | $1,350 |
| RDS (db.r6g.xlarge Multi-AZ) | $350 | $350 | $700 |
| ElastiCache (cache.r6g.xlarge) | $180 | $180 | $360 |
| MSK (3x kafka.m5.large) | $200 | $200 | $600 |
| S3 + CloudFront | $20 | $50 | $200 |
| CloudWatch / Grafana Cloud | $50 | $100 | $300 |
| **Total/Month** | **~$930** | **~$1,330** | **~$3,510** |

---

## 6. Team Structure

| Role | Count | Responsibilities |
|------|-------|------------------|
| **Tech Lead** | 1 | Architecture, code review, mentoring |
| **Backend Engineers** | 3 | API, services, ML pipeline, infra |
| **Frontend Engineer** | 2 | React, design system, testing |
| **ML Engineer** | 2 | Model training, registry, serving |
| **DevOps/Platform** | 2 | CI/CD, infra, security, observability |
| **QA/Automation** | 1 | Test automation, E2E, performance |
| **Product Manager** | 1 | Roadmap, stakeholders, compliance |
| **Security Engineer** | 1 (part-time) | Threat modeling, audits, compliance |

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ML model performance on unseen subjects | High | High | LOSO validation, few-shot calibration, ensemble |
| Data privacy regulation changes | Medium | High | Privacy by design, legal review quarterly |
| Key person dependency (ML expertise) | High | High | Documentation, pair programming, knowledge sharing |
| GPU cost overruns | Medium | Medium | Budget alerts, spot instances, auto-scaling |
| Third-party API dependencies | Low | High | Circuit breakers, fallback strategies |
| Regulatory approval delays | Medium | High | Early engagement, pre-submission meetings |

---

## 7. Milestones & Timeline

| Milestone | Target Date | Criteria |
|-----------|-------------|----------|
| **M0: Foundation Complete** | Week 2 | Repo, CI, dev env, standards |
| **M1: Core API Running** | Week 4 | Auth, Projects, Health checks |
| **M2: Training Pipeline Live** | Week 6 | Async jobs, model registry |
| **M3: Inference API Live** | Week 8 | Sync + batch, model serving |
| **M4: Frontend MVP** | Week 10 | Dashboard, training, inference UI |
| **M5: Security Hardened** | Week 12 | Pen test passed, audit ready |
| **M6: Observability Live** | Week 14 | Dashboards, alerts, runbooks |
| **M7: Production Deploy** | Week 16 | Staging → Canary → Prod |
| **M8: Compliance Ready** | Week 18 | SOC2 evidence, HIPAA assessment |
| **M9: GA Release** | Week 20 | v1.0.0 tagged, docs complete |

---

## 8. Definition of Done (per Story)

- [ ] Code complete, self-reviewed
- [ ] Unit tests ≥90% coverage
- [ ] Integration tests pass
- [ ] Contract tests pass
- [ ] E2E tests pass (critical paths)
- [ ] Security scan passes (zero critical/high)
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Peer reviewed (2 approvals)
- [ ] Deployed to staging, smoke tested
- [ ] Merged to main, semantic version bumped

---

## 9. Immediate Next Steps (This Week)

1. **Finalize `.gitignore`** and commit clean repo structure
2. **Set up Docker Compose** for local dev (Postgres, Redis, RabbitMQ, MinIO, Keycloak)
3. **Configure GitHub Actions** CI pipeline (lint → typecheck → test → build)
4. **Create FastAPI app factory** with DI, config, lifespan
5. **Add PostgreSQL + Alembic** with initial migration
6. **Implement Auth Service** (register, login, JWT, refresh)
6. **Configure Kong API Gateway** locally
7. **Add pytest + testcontainers** for integration tests
8. **Configure Ruff, Black, MyPy, Pre-commit**
9. **Set up dependabot + renovate** for dependency updates
10. **Create ADR template** and first ADR (architecture choice)

---

## 10. Appendix: Repository Structure (Target)

```
neuro_move_DRAFT/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── cd-staging.yml
│   │   ├── cd-production.yml
│   │   ├── dependabot.yml
│   │   └── security-scan.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── .vscode/
│   ├── settings.json
│   ├── extensions.json
│   └── launch.json
├── .github/
├── .pre-commit-config.yaml
├── .gitignore
├── .editorconfig
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── docker-compose.override.yml
├── docker-compose.prod.yml
├── Makefile
├── README.md
├── ENTERPRISE_PLAN.md
├── ARCHITECTURE.md
├── SECURITY.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── .env.example
├── .env.example.prod
├── infra/
│   ├── terraform/
│   └── kubernetes/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── security.py
│   │   ├── database.py
│   │   ├── cache.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── dependencies.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── project.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── training_job.py
│   │   ├── inference_job.py
│   │   └── audit.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── project.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   └── common.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── projects.py
│   │   │   ├── datasets.py
│   │   │   ├── training.py
│   │   │   ├── inference.py
│   │   │   ├── models.py
│   │   │   ├── audits.py
│   │   │   └── health.py
│   │   └── routes.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── project_service.py
│   │   ├── dataset_service.py
│   │   ├── training_service.py
│   │   ├── inference_service.py
│   │   ├── model_service.py
│   │   ├── audit_service.py
│   │   ├── ml_pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
│   │   │   └── registry.py
│   │   └── notification_service.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   ├── training_tasks.py
│   │   ├── inference_tasks.py
│   │   └── notification_tasks.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── correlation_id.py
│   │   ├── rate_limit.py
│   │   ├── auth.py
│   │   └── logging.py
│   └── utils/
│       ├── __init__.py
│       ├── validators.py
│       ├── helpers.py
│       └── constants.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_auth.py
│   │   ├── test_schemas.py
│   │   └── test_utils.py
│   ├── integration/
│   │   ├── test_auth_flow.py
│   │   ├── test_training_pipeline.py
│   │   └── test_inference.py
│   ├── contract/
│   │   └── test_api_contracts.py
│   ├── e2e/
│   │   ├── test_dashboard.spec.ts
│   │   └── test_training_flow.spec.ts
│   └── performance/
│       └── test_inference_latency.py
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── .eslintrc.json
│   ├── .prettierrc
│   ├── .storybook/
│   ├── public/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── routes.tsx
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   ├── forms/
│   │   │   ├── layout/
│   │   │   └── charts/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Projects.tsx
│   │   │   ├── Datasets.tsx
│   │   │   ├── Training.tsx
│   │   │   ├── Inference.tsx
│   │   │   ├── Models.tsx
│   │   │   └── Settings.tsx
│   │   ├── hooks/
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── auth.ts
│   │   │   └── websocket.ts
│   │   ├── store/
│   │   ├── types/
│   │   ├── utils/
│   │   └── styles/
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   └── .storybook/
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── guides/
│   ├── runbooks/
│   └── adr/
│       ├── 0001-record-architecture-decisions.md
│       ├── 0002-api-framework-choice.md
│       ├── 0003-database-choice.md
│       └── 0004-auth-strategy.md
└── scripts/
    ├── dev-setup.sh
    ├── db-migrate.sh
    ├── db-seed.sh
    ├── load-test.sh
    └── deploy.sh
```

---

## Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Tech Lead | | | |
| Engineering Manager | | | |
| Security Officer | | | |
| Product Owner | | | |
| Compliance Officer | | | |

---

*Document Version: 1.0*  
*Last Updated: 2026-10-18*  
*Classification: Internal - Confidential*