# Build context: repo root (needs streamhub/ and webui/).
FROM node:24-alpine AS webui
WORKDIR /src
COPY webui/package.json webui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY webui/ ./
RUN npm run build

FROM golang:1.27-alpine AS build
WORKDIR /src
COPY streamhub/go.mod streamhub/go.sum ./
RUN go mod download
COPY streamhub/cmd ./cmd
COPY streamhub/internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ ./cmd/streamhub ./cmd/pruner

FROM alpine:3
COPY --from=build /out/ /usr/local/bin/
COPY --from=webui /src/dist /webui
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8090/healthz || exit 1
ENTRYPOINT ["streamhub"]
CMD ["-config", "/config.yaml"]
