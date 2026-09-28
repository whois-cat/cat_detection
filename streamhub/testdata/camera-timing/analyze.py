import csv, gzip, sys
from collections import Counter
import numpy as np

def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    return list(csv.DictReader(opener(path, "rt")))

for name in sys.argv[2:]:
    rows = load(f"{sys.argv[1]}/{name}.rtp.csv.gz")
    # group into frames by rtp_ts (in arrival order); frame complete at marker
    frames = []  # (rtp_ts, first_arrival, last_arrival, nal types, size, seq_first)
    cur = None
    seq_gaps = 0
    prev_seq = None
    for r in rows:
        seq = int(r["seq"])
        if prev_seq is not None and (seq - prev_seq) % 65536 != 1:
            seq_gaps += 1
        prev_seq = seq
        ts = int(r["rtp_ts"]); a = int(r["arrival_ns"])
        if cur is None or cur["ts"] != ts:
            if cur: frames.append(cur)
            cur = {"ts": ts, "a0": a, "a1": a, "nals": set(), "size": 0, "marker": False}
        cur["a1"] = a; cur["nals"].add(r["nal"]); cur["size"] += int(r["size"])
        cur["marker"] |= r["marker"] == "1"
    frames.append(cur)
    # drop the initial connect burst: first 2 s of arrival
    ts = np.array([f["ts"] for f in frames], dtype=np.int64)
    ts = np.unwrap(ts, period=2**32).astype(np.int64) if hasattr(np, "unwrap") else ts
    arr = np.array([f["a1"] for f in frames], dtype=np.int64) / 1e9
    t_rtp = (ts - ts[0]) / 90000.0
    d_rtp = np.diff(t_rtp) * 1000
    d_arr = np.diff(arr) * 1000
    idr = [i for i, f in enumerate(frames) if "5" in f["nals"] or any(n.startswith("stap") and "5" in n for n in f["nals"])]
    sps_frames = [i for i, f in enumerate(frames) if any("7" in n for n in f["nals"])]
    other_nals = Counter(n for f in frames for n in f["nals"])
    dur = t_rtp[-1]
    print(f"===== {name}: {len(frames)} frames, rtp span {dur:.2f}s, arrival span {arr[-1]-arr[0]:.2f}s, "
          f"avg fps(rtp) {(len(frames)-1)/dur:.3f}, seq gaps {seq_gaps}, frames w/o marker {sum(not f['marker'] for f in frames)}")
    print(f"  NAL types seen: {dict(other_nals)}")
    gop = np.diff(idr)
    print(f"  IDR count {len(idr)}, GOP frames: {Counter(gop.tolist()).most_common(5)}; SPS-carrying frames {len(sps_frames)}")
    print(f"  RTP dt ms: min {d_rtp.min():.1f} p5 {np.percentile(d_rtp,5):.1f} median {np.median(d_rtp):.1f} "
          f"p95 {np.percentile(d_rtp,95):.1f} max {d_rtp.max():.1f} std {d_rtp.std():.1f}; non-increasing {int((d_rtp<=0).sum())}")
    print(f"  RTP dt histogram (ms, 5ms bins): {Counter((np.round(d_rtp/5)*5).astype(int).tolist()).most_common(8)}")
    print(f"  arrival dt ms: median {np.median(d_arr):.1f} p95 {np.percentile(d_arr,95):.1f} max {d_arr.max():.1f}; "
          f"<5ms (bursts) {int((d_arr<5).sum())} of {len(d_arr)}")
    big = np.where(d_rtp > 150)[0]
    print(f"  RTP gaps >150ms: {len(big)}", [(round(t_rtp[i],2), round(d_rtp[i])) for i in big[:12]])
    # latency jitter: arrival - rtp time (after start burst), linear fit for clock drift
    m = arr - arr[0] > 3
    lag = (arr - arr[0]) - t_rtp
    slope, icpt = np.polyfit(t_rtp[m], lag[m], 1)
    res = lag[m] - (slope * t_rtp[m] + icpt)
    print(f"  arrival-vs-rtp: drift {slope*1e6:+.0f} ppm (rtp clock vs local), residual std {res.std()*1000:.1f}ms, "
          f"p99 {np.percentile(np.abs(res),99)*1000:.0f}ms, max {np.abs(res).max()*1000:.0f}ms; "
          f"start burst: frames arriving in first 50ms: {int(((arr-arr[0])<0.05).sum())}")
    # RTCP
    rc = load(f"{sys.argv[1]}/{name}.rtcp.csv.gz")
    ntp = np.array([float(r["ntp_s"]) for r in rc]); rts = np.array([int(r["rtp_ts"]) for r in rc], dtype=np.int64)
    wall = np.array([int(r["wall_ns"]) for r in rc]) / 1e9
    if len(rc) > 2:
        s2, i2 = np.polyfit(rts - rts[0], ntp - ntp[0], 1)
        r2 = (ntp - ntp[0]) - (s2 * (rts - rts[0]) + i2)
        print(f"  RTCP SR: n={len(rc)}, NTP-vs-RTP rate {1/s2:.1f} Hz (nominal 90000), resid max {np.abs(r2).max()*1000:.1f}ms; "
              f"cam NTP - local wall: {np.median(ntp-wall)*1000:+.0f}ms (spread {np.ptp(ntp-wall)*1000:.0f}ms); "
              f"first SR rtp_ts {rts[0]} vs first frame ts {frames[0]['ts']}")
