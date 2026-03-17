
```bash
uv sync --all 

```

```bash
```

3. Follow https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/38 to remove the dependency

Edit your ~/.bashrc (or ~/.bash_profile on macOS):

```bash
nano ~/.bashrc
```

Add this line at the end:

```bash
export TRITON_PTXAS_PATH=$(which ptxas)
```


rpc
```bash
sudo systemctl enable rabbitmq-server
sudo systemctl start rabbitmq-server
sudo rabbitmq-plugins enable rabbitmq_management
```
firewall
```bash 
sudo ufw allow 8000/tcp # FastAPI
sudo ufw allow 15672/tcp # RabbitMQ Management
sudo ufw allow 5555/tcp # Celery Flower
sudo ufw reload 
sudo ufw status

```



## When error
```bash
export TRITON_PTXAS_PATH=$(which ptxas)
```
```bash
sudo cp /home/admin/Pre-trained/script/celery-flower.service /etc/systemd/system/celery-flower.service
sudo cp /home/admin/Pre-trained/script/gpt_worker.service /etc/systemd/system/gpt_worker.service
sudo cp /home/admin/Pre-trained/script/gpu_server.service /etc/systemd/system/gpu_server.service
sudo cp /home/admin/Pre-trained/script/ocr_worker.service /etc/systemd/system/ocr_worker.service


# Reload systemd manager configuration
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable celery-flower

sudo systemctl enable gpu_server

sudo systemctl enable ocr_worker
sudo systemctl enable gpt_worker
# Start services immediately
sudo systemctl start celery-flower
sudo systemctl start gpt_worker
sudo systemctl start gpu_server
sudo systemctl start ocr_worker

sudo systemctl restart celery-flower
sudo systemctl restart gpt_worker
sudo systemctl restart gpu_server
sudo systemctl restart ocr_worker
```


# Check status of each service
```bash
systemctl is-active celery-flower gpt_worker gpu_server rabbitmq-server ocr_worker
```


```bash 
sudo systemctl status celery-flower
sudo systemctl status ocr_worker
sudo systemctl status gpt_worker
sudo systemctl status gpu_server
sudo systemctl status rabbitmq-server
```

```bash
sudo journalctl -u gpu_server.service -f
sudo journalctl -u gpt_worker.service -f
sudo journalctl -u ocr_worker.service -f
sudo journalctl -u celery-flower.service -f

```

```bash 
sudo systemctl disable gpu_server
sudo systemctl stop gpu_server
sudo systemctl disable celery-flower
sudo systemctl stop celery-flower
sudo systemctl disable gpt_worker # GPT oss
sudo systemctl stop gpt_worker
sudo systemctl disable ocr_worker # deepseek 
sudo systemctl stop ocr_worker

```