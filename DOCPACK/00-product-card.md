---
title: solo-mlops — Product Card
doc_id: CARD-solo-mlops
doc_type: product-card
verdict: showcase
version: 0.1
date: 2026-10-11
status: ร่างเพื่อพิจารณา
owner: Product owner (Book Dev)
---

# solo-mlops — Product Card

## 1. สรุปคุณค่าในหนึ่งประโยค

ตัวอย่าง pipeline MLOps ครบวง พร้อม gold dataset สังเคราะห์และ model release `solo-empire-central-assistant` เพื่อพิสูจน์แนวทาง train-to-monitor

## 2. ปัญหาและผู้ใช้เป้าหมาย

ต้องมี reference ว่า model lifecycle ควรตรวจอะไรบ้าง ผู้ใช้คือเจ้าของ (เรียนรู้/เตรียมงาน MLOps)

## 3. สิ่งที่ทำได้จริงตอนนี้

- โค้ดและเทสต์ใน `infra/mlops`, `infra/scripts/mlops`, `infra/tests/mlops` (4 ไฟล์) และ 1 CI workflow
- รันเทสต์ด้วย `python3 infra/tests/mlops/mlops_pipeline_test.py`

## 4. หลักฐานสถานะ

| รายการ | ค่า |
|---|---|
| Stack | Python |
| Commit | 19 ครั้ง (2026-08-08 ถึง 2026-10-11) |
| ซอร์สโค้ด | 14 ไฟล์ / 2874 บรรทัด |
| เทสต์ | 4 ไฟล์ (ไม่ได้รันในงานนี้) |
| CI | 1 workflow |
| Deploy config | ไม่มี |
| Live URL | ไม่มี |

## 5. คุณค่าและจุดต่าง

- แสดงทักษะ MLOps แบบมี gate ตรวจสอบ
- **[สมมติฐาน]** ไม่มีมูลค่าขาย — ข้อมูลเป็นแบบสังเคราะห์

## 6. ช่องว่างสำคัญ

- ข้อมูลเป็น synthetic ไม่ได้ train โมเดลธุรกิจจริง
- config Prometheus/Grafana อยู่ที่ solo-homelab

## 7. คำตัดสิน

**`showcase`** — หลักฐานทักษะและ reference ที่ดี ไม่ใช่ผลิตภัณฑ์ จะเปลี่ยนเป็น `invest` หากมี use case โมเดลจริงที่ต้องใช้ pipeline นี้

## 8. ขั้นต่อไป

1. รัน pipeline test และบันทึกผลใน README

## การอนุมัติ

| บทบาท | สถานะ | วันที่ | ลงนาม |
|---|---|---|---|
| Product owner (Book Dev) | รอพิจารณา | — | ________ |

## แหล่งข้อมูล

- `README.md`
- `infra/mlops/`, `infra/tests/mlops/`
