# Docker Setup for CareerPath AI

This folder contains all Docker configuration files to run CareerPath AI in containers.

## Quick Start

### 1. Build and Run

**Important:** Run from the **project root directory** for best compatibility:

```bash
cd careerpath-ai-main
docker-compose -f docker/docker-compose.yml up --build
```

Or to run in the background:

```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

### 2. Access the Application

- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Stop the Application

```bash
docker-compose -f docker/docker-compose.yml down
```

## Important Notes

### Running from Project Root

**Always run docker-compose from the project root directory** (where `backend/` and `frontend/` folders are located), not from inside the `docker/` folder. This ensures proper path resolution for the build context.

### Data Persistence

The trained ML models and datasets are **included in the Docker image** during build (not mounted as a volume). This approach ensures compatibility across different operating systems, especially Windows with Docker Desktop.

If you need to update the models:
1. Update the files in `backend/data/`
2. Rebuild the images: `docker-compose -f docker/docker-compose.yml up --build`

## What's Included

```
docker/
├── backend/
│   ├── Dockerfile           # Backend container configuration
│   └── .dockerignore        # Files to exclude from backend image
├── frontend/
│   ├── Dockerfile           # Frontend container configuration
│   ├── nginx.conf           # Nginx web server configuration
│   └── .dockerignore        # Files to exclude from frontend image
├── docker-compose.yml       # Multi-container orchestration
└── README.md               # This file
```

## Services

### Backend (FastAPI)
- **Port**: 8000
- **Base Image**: Python 3.11-slim
- **Features**: 
  - FastAPI REST API
  - Career path prediction model
  - PDF text extraction
  - Data persistence via volume mount

### Frontend (React + Vite)
- **Port**: 80
- **Base Image**: Node 18 (build) + Nginx Alpine (serve)
- **Features**:
  - React single-page application
  - Nginx reverse proxy for API requests
  - Gzip compression
  - Static asset caching
  - React Router support

## Commands

### Build Images
```bash
# Build all services
docker-compose up --build

# Build without starting
docker-compose build

# Build specific service
docker-compose build backend
docker-compose build frontend
```

### Run Services
```bash
# Run in foreground
docker-compose up

# Run in background (detached)
docker-compose up -d

# Run specific service
docker-compose up backend
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Manage Containers
```bash
# List running containers
docker-compose ps

# Stop services
docker-compose stop

# Restart services
docker-compose restart

# Remove containers and networks
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v
```

## Sharing to Other Machines

### Method 1: Share the Entire Project
1. Copy the entire `careerpath-ai` folder to the target machine
2. Navigate to the `docker` folder
3. Run: `docker-compose up --build`

### Method 2: Export/Import Docker Images

**On your machine:**
```bash
# Build images
docker-compose build

# Save images to tar files
docker save careerpath-ai-backend:latest | gzip > careerpath-backend.tar.gz
docker save careerpath-ai-frontend:latest | gzip > careerpath-frontend.tar.gz
```

**On target machine:**
```bash
# Load images
docker load < careerpath-backend.tar.gz
docker load < careerpath-frontend.tar.gz

# Run with docker-compose
docker-compose up
```

### Method 3: Docker Hub (Public/Private Registry)

**Push to Docker Hub:**
```bash
# Login
docker login

# Tag images
docker tag careerpath-ai-backend:latest yourusername/careerpath-ai-backend:latest
docker tag careerpath-ai-frontend:latest yourusername/careerpath-ai-frontend:latest

# Push images
docker push yourusername/careerpath-ai-backend:latest
docker push yourusername/careerpath-ai-frontend:latest
```

**Pull on target machine:**
```bash
docker pull yourusername/careerpath-ai-backend:latest
docker pull yourusername/careerpath-ai-frontend:latest
docker-compose up
```

## Configuration

### Change Ports

Edit `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8001:8000"  # External:Internal
  frontend:
    ports:
      - "8080:80"    # External:Internal
```

### Environment Variables

Create a `.env` file in the docker directory:

```env
BACKEND_PORT=8000
FRONTEND_PORT=80
```

Then update `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "${BACKEND_PORT}:8000"
```

## Troubleshooting

### Port Already in Use
If you get "port already in use" errors:
- Change the external port in `docker-compose.yml`
- Or stop the process using that port

**Windows:**
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
lsof -i :8000
kill -9 <PID>
```

### Backend Not Starting
```bash
# Check logs
docker-compose logs backend

# Check if dependencies are installed
docker-compose exec backend pip list

# Rebuild without cache
docker-compose build --no-cache backend
```

### Frontend Can't Connect to Backend
- Ensure both services are running: `docker-compose ps`
- Check backend health: http://localhost:8000/docs
- Verify nginx configuration in `frontend/nginx.conf`

### Permission Errors (Linux/Mac)
```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ../backend/data
```

### Clear Everything and Start Fresh
```bash
# Stop and remove all containers, networks, and volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Rebuild from scratch
docker-compose up --build
```

## System Requirements

- **Docker Engine**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Disk Space**: 2GB for images and data

## Health Checks

The backend includes a health check that verifies the service is running:
- Endpoint: `/docs`
- Interval: 30 seconds
- Timeout: 10 seconds
- Start period: 40 seconds

Check health status:
```bash
docker inspect careerpath-ai-backend --format='{{.State.Health.Status}}'
```

## Network Architecture

Both services communicate through a dedicated Docker network (`careerpath-network`):
- Frontend → Backend: via nginx proxy at `/api`
- Backend → Frontend: Not needed
- External → Frontend: Port 80
- External → Backend: Port 8000

## Data Persistence

The backend's data directory is mounted as a volume:
- **Host**: `../backend/data`
- **Container**: `/app/data`
- **Purpose**: Persist trained ML models and datasets

This ensures your models survive container restarts and rebuilds.

## Production Considerations

For production deployments:
1. Use specific version tags instead of `latest`
2. Set up proper secrets management
3. Configure SSL/TLS certificates
4. Set up proper logging and monitoring
5. Use Docker Swarm or Kubernetes for orchestration
6. Implement backup strategies for volumes
7. Configure resource limits (CPU, memory)

Example resource limits:
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

## Support

For issues or questions:
1. Check the logs: `docker-compose logs -f`
2. Verify all files are present in the docker folder
3. Ensure Docker and Docker Compose are up to date
4. Review the main project documentation

## License

See the main project README for license information.
