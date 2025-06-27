FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install curl for healthchecks
RUN apk add --no-cache curl

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Install serve to run the built app
RUN npm install -g serve

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/ || exit 1

# Command to serve the built application
CMD ["serve", "-s", "build", "-l", "3000"]
