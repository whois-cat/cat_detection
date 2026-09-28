// Command pruner sparsifies and caps recordings: old segments without
// detections nearby are deleted, then the oldest go until the total fits the
// size cap. Pinned ranges and recent recordings are kept (see internal/prune).
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/whois-cat/cat_detection/streamhub/internal/config"
	"github.com/whois-cat/cat_detection/streamhub/internal/pins"
	"github.com/whois-cat/cat_detection/streamhub/internal/prune"
)

func main() {
	configPath := flag.String("config", "config.yaml", "config file")
	once := flag.Bool("once", false, "run one pass and exit")
	dryRun := flag.Bool("dry-run", false, "only log what would be deleted (also pruner.dry_run)")
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error("config", "err", err)
		os.Exit(1)
	}
	pc := cfg.Pruner
	policy := prune.Policy{
		KeepRecent:        time.Duration(pc.KeepRecent),
		EventMargin:       time.Duration(pc.EventMargin),
		DeleteUnprocessed: pc.DeleteUnprocessed,
		MaxBytes:          int64(pc.MaxSize),
	}
	dry := *dryRun || pc.DryRun
	log.Info("pruner started", "policy", policy, "interval", time.Duration(pc.Interval), "dry_run", dry)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	for {
		pass(cfg, policy, dry, log)
		if *once {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Duration(pc.Interval)):
		}
	}
}

func pass(cfg *config.Config, policy prune.Policy, dry bool, log *slog.Logger) {
	root := cfg.RecordingsDir()
	ps, err := pins.Load(cfg.PinsFile())
	if err != nil {
		// Without pins we could delete something the user wants kept.
		log.Error("reading pins; skipping pass", "err", err)
		return
	}
	cams, err := prune.Scan(root)
	if err != nil {
		log.Error("scanning recordings; skipping pass", "err", err)
		return
	}
	var total int64
	var n int
	for _, ss := range cams {
		for _, s := range ss {
			total += s.Bytes
			n++
		}
	}
	dels := prune.Plan(cams, ps, policy, time.Now())
	freed, err := prune.Apply(root, dels, dry, log)
	if err != nil {
		log.Error("deleting", "err", err)
	}
	log.Info("pass done", "segments", n, "bytes", total, "deleted", len(dels), "freed", freed, "dry_run", dry)
}
