# Production — build Angular + serve via Nginx
FROM node:22-alpine AS builder

WORKDIR /app

RUN npm install -g @angular/cli@latest --quiet

COPY package*.json ./
RUN npm install

COPY . .
RUN ng build --configuration production

# --- Runtime ---
FROM nginx:alpine

COPY --from=builder /app/dist/lucy-frontend/browser /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
