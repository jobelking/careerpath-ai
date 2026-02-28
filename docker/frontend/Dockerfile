# Frontend Dockerfile for CareerPath AI
# Build stage
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY frontend .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration as a template
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf.template

# Expose port 80
EXPOSE 80

# Substitute env vars into nginx config then start nginx
# BACKEND_URL defaults to http://backend:8000 for Docker Compose local use
CMD ["/bin/sh", "-c", "export BACKEND_URL=$(echo ${BACKEND_URL:-http://backend:8000} | sed 's|/$||'); envsubst '$BACKEND_URL' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
