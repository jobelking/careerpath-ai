# 🚀 Quick Start Guide

## Your Docker Setup is Ready!

All Docker files are organized in the `docker/` folder for easy sharing.

### ✅ What's Running

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend App**: http://localhost

### 📦 Files Created

```
docker/
├── backend/
│   ├── Dockerfile          # Backend Python container
│   └── .dockerignore       # Excludes unnecessary files
├── frontend/
│   ├── Dockerfile          # Frontend React + Nginx container
│   ├── nginx.conf          # Web server configuration
│   └── .dockerignore       # Excludes unnecessary files
├── docker-compose.yml      # Orchestration configuration
├── README.md              # Detailed documentation
└── QUICKSTART.md          # This file
```

### 🎯 Basic Commands

**Important:** Always run from the **project root directory** (where `backend/` and `frontend/` folders are):

```bash
# Navigate to project root
cd careerpath-ai-main

# Start services
    docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml stop

# Remove everything
docker-compose -f docker/docker-compose.yml down

# Rebuild and start
docker-compose -f docker/docker-compose.yml up --build
```

### 📤 Sharing to Another Machine

**Method 1: Share the Project Folder**
1. Copy the entire `careerpath-ai` folder
2. On the target machine, navigate to `docker/` folder
3. Run: `docker-compose up --build`

**Method 2: Export/Import Images**

On your machine:
```bash
cd docker
docker-compose build
docker save docker-backend:latest | gzip > careerpath-backend.tar.gz
docker save docker-frontend:latest | gzip > careerpath-frontend.tar.gz
```

On target machine:
```bash
docker load < careerpath-backend.tar.gz
docker load < careerpath-frontend.tar.gz
docker-compose up
```

### 🔧 Troubleshooting

**Port Conflicts:**
Edit `docker-compose.yml` to change ports:
```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Change 8000 to any available port
  frontend:
    ports:
      - "8080:80"    # Change 80 to any available port
```

**Check Health Status:**
```bash
docker-compose ps
docker-compose logs backend
docker-compose logs frontend
```

### 📖 More Information

See the complete [docker/README.md](docker/README.md) for:
- Detailed configuration options
- Production deployment tips
- Advanced troubleshooting
- Network architecture
- Security considerations

---

**Your application is now containerized and ready to share!** 🎊
