# NeuroMove Enterprise MI-BCI - Deployment Guide

## Overview

Production-ready deployment architecture for NeuroMove Enterprise MI-BCI platform with:
- **FastAPI** REST + WebSocket inference server
- **Streamlit** enterprise dashboard
- **MLflow** experiment tracking & model registry
- **Prometheus + Grafana** observability stack
- **PostgreSQL** metadata store
- **Redis** caching layer
- **Docker** containerization with security hardening

---

## Quick Start

### Prerequisites
- Docker 24+ & Docker Compose 2.20+
- 8GB+ RAM, 4+ CPU cores
- NVIDIA GPU (optional, for CUDA acceleration)
- 50GB+ disk space

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

Required variables:
```env
# API Security
API_KEY_ADMIN=your-secure-admin-key
API_KEY_CLIENT=your-secure-client-key

# Database
POSTGRES_PASSWORD=your-secure-postgres-password

# Grafana
GRAFANA_PASSWORD=your-secure-grafana-password

# Alerting (optional)
SMTP_USERNAME=your-smtp-user
SMTP_PASSWORD=your-smtp-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PAGERDUTY_KEY=your-pagerduty-key
```

### Deploy

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f neuromove-api
docker-compose logs -f neuromove-dashboard
```

### Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | API Key auth |
| Dashboard | http://localhost:8501 | - |
| MLflow | http://localhost:5000 | - |
| Grafana | http://localhost:3000 | admin / GRAFANA_PASSWORD |
| Prometheus | http://localhost:9090 | - |
| Alertmanager | http://localhost:9093 | - |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐        ┌──────────┐      ┌─────────┐
   │  API    │        │Dashboard │      │ MLflow  │
   │ :8000   │        │ :8501    │      │ :5000   │
   └────┬────┘        └────┬─────┘      └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
        ┌─────────────────────────────────┐
        │      Shared Volumes             │
        │  Models / Artifacts / MLflow    │
        └─────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐        ┌──────────┐      ┌─────────┐
   │PostgreSQL│       │  Redis   │      │Prometheus│
   │  :5432   │       │  :6379   │      │  :9090   │
   └────┬────┘        └────┬─────┘      └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    ┌──────────────┐
                    │   Grafana    │
                    │   :3000      │
                    └──────────────┘
```

---

## Service Details

### NeuroMove API (`neuromove-api`)

**Endpoints:**
- `GET /health` - Health check
- `GET /ready` - Readiness probe
- `GET /metrics` - JSON metrics (auth required)
- `GET /metrics/prometheus` - Prometheus format
- `POST /inference` - Batch inference
- `WS /ws/inference` - Real-time streaming
- `GET /models` - List registered models
- `POST /models/load` - Load model
- `POST /models/unload` - Unload model

**Authentication:** Bearer token (API Key)
```bash
curl -H "Authorization: Bearer neuromove-enterprise" \
     -H "Content-Type: application/json" \
     -d '{"data": [[[...]]]}' \
     http://localhost:8000/inference
```

**WebSocket Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/inference');
ws.onopen = () => ws.send(JSON.stringify({data: eegData, request_id: '123'}));
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

### NeuroMove Dashboard (`neuromove-dashboard`)

Streamlit-based enterprise dashboard with:
- 11 tabs: Overview, Model Architectures, Live Training, Live Inference, etc.
- Real-time EEG visualization
- Model comparison & benchmarking
- Configuration-driven via `config.yaml`

### MLflow (`mlflow`)

Experiment tracking at `http://localhost:5000`:
- Experiment management
- Run comparison
- Model registry with versioning
- Artifact storage

**Python SDK:**
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("neuromove-training")

with mlflow.start_run():
    mlflow.log_param("num_kernels", 10000)
    mlflow.log_metric("accuracy", 0.9863)
    mlflow.sklearn.log_model(pipeline, "model")
```

### Monitoring Stack

**Prometheus** (`:9090`): Metrics collection
- Scrapes API `/metrics/prometheus` every 10s
- Custom rules in `monitoring/rules/neuromove.yml`

**Grafana** (`:3000`): Dashboards
- Pre-configured NeuroMove dashboard
- Auto-provisioned from `monitoring/grafana-provisioning/`

**Alertmanager** (`:9093`): Alerting
- Email, Slack, PagerDuty integrations
- Config in `monitoring/alertmanager.yml`

---

## Model Management

### Register Model via API
```bash
curl -X POST http://localhost:8000/models/load \
  -H "Authorization: Bearer neuromove-enterprise" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "minirocket_latest"}'
```

### Register Model via MLflow
```python
from mlflow.mlflow_tracker import NeuroMoveMLflowTracker, TrainingRun

tracker = NeuroMoveMLflowTracker()

with TrainingRun(tracker, "minirocket_physionet", "minirocket", "physionet", "1-109") as run:
    run.log_params({"num_kernels": 10000})
    run.log_metrics({"accuracy": 0.9863})
    run.log_model(pipeline, "minirocket_physionet", 0.9863)
```

### Load Best Model
```python
from mlflow.mlflow_tracker import get_best_model, load_model_from_registry

best = get_best_model("minirocket", "physionet")
model = load_model_from_registry("minirocket", "physionet")
```

---

## Security Hardening

### Network
- All services on isolated Docker network (`172.28.0.0/16`)
- Only API (`8000`), Dashboard (`8501`), Grafana (`3000`) exposed
- Internal services (PostgreSQL, Redis, MLflow) not exposed

### Authentication
- API: Bearer token authentication
- MLflow: No auth (add proxy for production)
- Grafana: Admin password from env

### Container Security
- Non-root user (`neuromove:1000`)
- Read-only config mounts
- Dropped capabilities
- Resource limits (CPU/Memory)

### Secrets Management
- All secrets via environment variables
- `.env` file (gitignored)
- Production: Use Docker secrets / Vault / AWS Secrets Manager

---

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.override.yml
services:
  neuromove-api:
    deploy:
      replicas: 3
    # Add load balancer (nginx/traefik) in front
```

### GPU Support
```yaml
services:
  neuromove-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Backup & Recovery

### Models & Artifacts
```bash
# Backup
docker run --rm -v neuromove_models:/data -v $(pwd):/backup alpine tar czf /backup/models_backup.tar.gz /data

# Restore
docker run --rm -v neuromove_models:/data -v $(pwd):/backup alpine tar xzf /backup/models_backup.tar.gz -C /
```

### MLflow Database
```bash
# Backup
docker exec neuromove-postgres pg_dump -U neuromove neuromove > mlflow_backup.sql

# Restore
docker exec -i neuromove-postgres psql -U neuromove neuromove < mlflow_backup.sql
```

### Grafana Dashboards
```bash
# Export via Grafana UI or API
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  http://localhost:3000/api/dashboards/uid/neuromove-dashboard > backup.json
```

---

## Troubleshooting

### Common Issues

**API won't start - model not found**
```bash
# Check model registry
curl -H "Authorization: Bearer neuromove-enterprise" http://localhost:8000/models
```

**High latency**
```bash
# Check GPU utilization
docker exec neuromove-api nvidia-smi

# Check model size
ls -lh models/
```

**MLflow connection refused**
```bash
# Check MLflow logs
docker-compose logs mlflow

# Verify backend store
docker exec neuromove-mlflow ls -la /app/mlruns
```

**Dashboard blank**
```bash
# Check Streamlit logs
docker-compose logs neuromove-dashboard

# Verify config mount
docker exec neuromove-dashboard cat /app/config/config.yaml
```

### Logs
```bash
# All services
docker-compose logs -f --tail=100

# Specific service
docker-compose logs -f neuromove-api

# JSON logs (structured)
docker-compose exec neuromove-api tail -f /app/logs/neuromove_api.json
```

---

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
- name: Deploy to Production
  run: |
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    docker-compose exec neuromove-api curl -f http://localhost:8000/ready
```

### Health Checks in Pipeline
```bash
# Pre-deploy
docker-compose build neuromove-api
docker-compose run --rm neuromove-api python -m pytest tests/

# Post-deploy
curl -f http://prod-api:8000/ready
```

---

## Compliance & Auditing

### HIPAA Considerations
- All PHI encrypted at rest (PostgreSQL TDE)
- TLS 1.3 for all inter-service communication
- Audit logs in `/app/logs/neuromove_api.json`
- Access control via API keys

### FDA Class II Readiness
- Model versioning via MLflow
- Immutable audit trail
- Reproducible training via config + MLflow
- V&V documentation generation

---

## Support

- **Documentation**: `/docs`
- **Issues**: GitHub Issues
- **Security**: security@neuromove.ai
- **Enterprise**: sales@neuromove.ai