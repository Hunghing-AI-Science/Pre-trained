# ============================================================
# Makefile — Pre-trained GPU Server
# ============================================================
PYTHON   ?= python
PROJECT  ?= ./
IMAGE    ?= /home/admin/Pre-trained/src/vllm/1752049668_7801_region_det_res.png
PROMPT   ?= Free OCR.
BASE_URL ?= http://localhost:8000

SERVICES := gpu_server gpt_worker vllm_ocr_worker celery-flower

.PHONY: help test-all \
        install deploy \
        start stop restart status log enable disable

# ── Default target ─────────────────────────────────────────────
help:
	@echo ""
	@echo "  Usage: make <target>"
	@echo ""
	@echo "  ── Tests ───────────────────────────────────────────"
	@echo "  test-all        Run all unit tests (no GPU / broker needed)"
	@echo ""
	@echo "  ── Deploy ──────────────────────────────────────────"
	@echo "  install         Copy all .service files to /etc/systemd/system/ and daemon-reload"
	@echo "  deploy          install + enable + start all services"
	@echo ""
	@echo "  ── Service control (all 5 services) ────────────────"
	@echo "  start           sudo systemctl start   all services"
	@echo "  stop            sudo systemctl stop    all services"
	@echo "  restart         sudo systemctl restart all services"
	@echo "  status          systemctl status       all services"
	@echo "  enable          sudo systemctl enable  all services"
	@echo "  disable         sudo systemctl disable all services"
	@echo "  log             Follow journals for all services"
	@echo ""
	@echo "  Services: $(SERVICES)"
	@echo ""

# ── Tests ──────────────────────────────────────────────────────

test-all:
	cd $(PROJECT) && $(PYTHON) test/test_vllm_ocr_service.py
	cd $(PROJECT) && $(PYTHON) test/test_vllm_ocr_task.py

# ── Systemd deploy ─────────────────────────────────────────────

SCRIPT_DIR := /home/admin/Pre-trained/script
SYSTEMD_DIR := /etc/systemd/system

install:
	sudo cp $(SCRIPT_DIR)/gpu_server.service        $(SYSTEMD_DIR)/gpu_server.service
	sudo cp $(SCRIPT_DIR)/gpt_worker.service        $(SYSTEMD_DIR)/gpt_worker.service
	sudo cp $(SCRIPT_DIR)/vllm_ocr_worker.service   $(SYSTEMD_DIR)/vllm_ocr_worker.service
	sudo cp $(SCRIPT_DIR)/celery-flower.service     $(SYSTEMD_DIR)/celery-flower.service
	sudo systemctl daemon-reload
	@echo "✓ Service files installed and daemon reloaded."

deploy: install enable start
	@echo "✓ All services deployed."

# ── Per-action targets ─────────────────────────────────────────

start:
	@for svc in $(SERVICES); do sudo systemctl start $$svc; done
	@echo "✓ All services started."

stop:
	@for svc in $(SERVICES); do sudo systemctl stop $$svc; done
	@echo "✓ All services stopped."

restart:
	@for svc in $(SERVICES); do sudo systemctl restart $$svc; done
	@echo "✓ All services restarted."

status:
	@systemctl status $(SERVICES) --no-pager

enable:
	@for svc in $(SERVICES); do sudo systemctl enable $$svc; done
	@echo "✓ All services enabled."

disable:
	@for svc in $(SERVICES); do sudo systemctl disable $$svc; done
	@echo "✓ All services disabled."

log:
	@echo "Tailing journals for: $(SERVICES)"
	sudo journalctl -f $(foreach svc,$(SERVICES),-u $(svc).service)
