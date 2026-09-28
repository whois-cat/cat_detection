// Command streamhub ingests camera streams, records them and serves them.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/api"
	"github.com/whois-cat/cat_detection/streamhub/internal/config"
	"github.com/whois-cat/cat_detection/streamhub/internal/index"
	"github.com/whois-cat/cat_detection/streamhub/internal/ingest"
	"github.com/whois-cat/cat_detection/streamhub/internal/media"
	"github.com/whois-cat/cat_detection/streamhub/internal/recorder"
	"github.com/whois-cat/cat_detection/streamhub/internal/segment"
)

// recorderBuffer is how many frames a recorder may lag behind ingest before
// frames are dropped (~10 s at 30 fps).
const recorderBuffer = 300

const cachedirTag = `Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by streamhub.
# Recordings are large and not worth backing up; set
# streamhub.recordings.cachedir_tag: false in config.yaml to change that.
# For information about cache directory tags, see https://bford.info/cachedir/
`

func main() {
	configPath := flag.String("config", "config.yaml", "config file")
	debug := flag.Bool("debug", false, "debug logging")
	flag.Parse()

	level := slog.LevelInfo
	if *debug {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	if err := run(*configPath, log); err != nil {
		log.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run(configPath string, log *slog.Logger) error {
	cfg, err := config.Load(configPath)
	if err != nil {
		return err
	}
	root := cfg.RecordingsDir()
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	if *cfg.Streamhub.Recordings.CachedirTag {
		if err := ensureFile(filepath.Join(root, "CACHEDIR.TAG"), cachedirTag); err != nil {
			return err
		}
	}
	if err := recorder.Recover(root, log); err != nil {
		return err
	}
	idx := index.New(root, log)
	if err := idx.Scan(); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	var wg sync.WaitGroup
	// fatal receives the first error that must bring the service down.
	fatal := make(chan error, len(cfg.Cameras)+1)
	sources := map[string]*ingest.Source{}
	target := time.Duration(cfg.Streamhub.Recordings.SegmentTarget)
	for _, cam := range cfg.Cameras {
		stream := media.NewStream()
		src := ingest.NewSource(cam.ID, cam.RTSP, stream, log)
		sources[cam.ID] = src
		rec := recorder.New(cam.ID, root, target, func(info segment.Info, size int64) {
			idx.Add(info, size)
			log.Debug("segment finished", "path", info.Path, "duration", info.Duration, "size", size)
		}, log)
		sub := stream.Subscribe(recorderBuffer)
		wg.Go(func() { src.Run(ctx) })
		wg.Go(func() {
			if err := rec.Run(ctx, sub); err != nil {
				// Losing recording is fatal: restart the whole service.
				fatal <- fmt.Errorf("recorder %s: %w", cam.ID, err)
				stop()
			}
		})
	}
	wg.Go(func() { idx.RescanEvery(ctx, time.Duration(cfg.Streamhub.Recordings.Rescan)) })

	srv := &http.Server{
		Addr: cfg.Streamhub.Listen,
		Handler: (&api.Server{
			Cameras: cfg.Cameras, Sources: sources, Index: idx, Root: root, Log: log,
		}).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			fatal <- fmt.Errorf("http: %w", err)
			stop()
		}
	}()
	log.Info("streamhub started", "listen", cfg.Streamhub.Listen, "cameras", len(cfg.Cameras), "recordings", root)

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Shutdown(shutdownCtx)
	wg.Wait()
	log.Info("streamhub stopped")
	select {
	case err := <-fatal:
		return err
	default:
		return nil
	}
}

// ensureFile creates path with content unless it exists.
func ensureFile(path, content string) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if errors.Is(err, os.ErrExist) {
		return nil
	}
	if err != nil {
		return err
	}
	_, err = f.WriteString(content)
	return errors.Join(err, f.Close())
}
