FROM golang:1.27-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /streamhub ./cmd/streamhub

FROM alpine:3
COPY --from=build /streamhub /usr/local/bin/streamhub
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8090/healthz || exit 1
ENTRYPOINT ["streamhub"]
CMD ["-config", "/config.yaml"]
