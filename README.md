# solo-mlops

Reference MLOps pipeline (dataset contract → train → evaluate → registry → staging → serve → monitor) พร้อม gold dataset สังเคราะห์และ model release `solo-empire-central-assistant`.

ที่มา: ย้ายจาก `solo-empire` เมื่อ 2026-10-11 พร้อมประวัติ git ของ `infra/mlops`, `infra/scripts/mlops`, `infra/tests/mlops`, `infra/tests/fixtures/mlops` (โครงสร้าง `infra/...` เดิมคงไว้).
สถาปัตยกรรม/ระบบ: `systems/ai-data/central-mlops-devsecops-pipeline.md` ใน solo-empire.
Prometheus/Grafana config ของ MLOps อยู่ใน `solo-homelab` (พร้อม `mlops_observability_test.py`).

## ทดสอบ

```bash
python3 infra/tests/mlops/mlops_pipeline_test.py
```
