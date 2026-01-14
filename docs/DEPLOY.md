


```bash
uv sync --all 

```

Edit your ~/.bashrc (or ~/.bash_profile on macOS):

```bash
nano ~/.bashrc
```

Add this line at the end:

```bash
export TRITON_PTXAS_PATH=$(which ptxas)
```

```bash 
sudo ufw allow 8000/tcp # FastAPI
sudo ufw allow 15672/tcp # RabbitMQ Management
sudo ufw allow 5555/tcp # Celery Flower
sudo ufw reload 
sudo ufw status

```
```bash
celery -A src.gp_server.u.celery_app worker

```

