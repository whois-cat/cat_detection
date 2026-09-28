"""Minimal RTSP-over-TCP client logging raw RTP/RTCP timing (no ffmpeg in the way).

usage: rtpcap.py <name> <rtsp_url> <seconds> <outdir>

Writes:
  <name>.rtp.csv   one row per RTP packet: arrival, seq, rtp_ts, marker, nal type, size
  <name>.rtcp.csv  one row per RTCP Sender Report: arrival, ntp time, rtp_ts
  <name>.h264      Annex-B elementary stream (playable with ffplay -f h264)
"""
import base64
import csv
import hashlib
import re
import socket
import struct
import sys
import time
from urllib.parse import urlsplit

name, url, seconds, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
u = urlsplit(url)
user, password = u.username, u.password
clean_url = f"rtsp://{u.hostname}:{u.port or 554}{u.path}"

sock = socket.create_connection((u.hostname, u.port or 554), timeout=10)
buf = b""
cseq = 0
auth_header = None


def recv_more():
    global buf
    chunk = sock.recv(65536)
    if not chunk:
        raise EOFError("connection closed")
    buf += chunk


def read_response():
    global buf
    while b"\r\n\r\n" not in buf:
        recv_more()
    head, buf = buf.split(b"\r\n\r\n", 1)
    lines = head.decode().split("\r\n")
    status = int(lines[0].split()[1])
    hdrs = {}
    for line in lines[1:]:
        k, v = line.split(":", 1)
        hdrs[k.strip().lower()] = v.strip()
    body = b""
    n = int(hdrs.get("content-length", 0))
    while len(buf) < n:
        recv_more()
    body, buf = buf[:n], buf[n:]
    return status, hdrs, body


def make_auth(method, uri, www):
    if www.lower().startswith("digest"):
        p = dict(re.findall(r'(\w+)="([^"]*)"', www))
        ha1 = hashlib.md5(f"{user}:{p['realm']}:{password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        resp = hashlib.md5(f"{ha1}:{p['nonce']}:{ha2}".encode()).hexdigest()
        return (f'Digest username="{user}", realm="{p["realm"]}", nonce="{p["nonce"]}", '
                f'uri="{uri}", response="{resp}"')
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


www_auth = None


def request(method, uri, extra=None):
    global cseq
    for _ in range(2):
        cseq += 1
        hdrs = {"CSeq": str(cseq), "User-Agent": "rtpcap"}
        if www_auth:
            hdrs["Authorization"] = make_auth(method, uri, www_auth)
        hdrs.update(extra or {})
        msg = f"{method} {uri} RTSP/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items()) + "\r\n"
        sock.sendall(msg.encode())
        status, rh, body = read_response()
        if status == 401 and not www_auth:
            set_auth(rh["www-authenticate"])
            continue
        if status != 200:
            raise RuntimeError(f"{method} -> {status}")
        return rh, body
    raise RuntimeError(f"{method}: auth failed")


def set_auth(v):
    global www_auth
    www_auth = v


rh, sdp = request("DESCRIBE", clean_url, {"Accept": "application/sdp"})
sdp = sdp.decode()
base = rh.get("content-base", clean_url + "/")
# first video media control
video = sdp[sdp.index("m=video"):]
nxt = video.find("\nm=", 1)
video = video if nxt < 0 else video[:nxt]
control = re.search(r"a=control:(\S+)", video).group(1)
track_url = control if control.startswith("rtsp://") else base.rstrip("/") + "/" + control
sprop = re.search(r"sprop-parameter-sets=([^;\s]+)", video)

rh, _ = request("SETUP", track_url, {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1"})
session = rh["session"].split(";")[0]
request("PLAY", base, {"Session": session, "Range": "npt=0.000-"})

with open(f"{outdir}/{name}.sdp", "w") as f:
    f.write(sdp)

h264 = open(f"{outdir}/{name}.h264", "wb")
if sprop:
    for ps in sprop.group(1).split(","):
        h264.write(b"\x00\x00\x00\x01" + base64.b64decode(ps))

rtp_f = open(f"{outdir}/{name}.rtp.csv", "w", newline="")
rtcp_f = open(f"{outdir}/{name}.rtcp.csv", "w", newline="")
rtp_w, rtcp_w = csv.writer(rtp_f), csv.writer(rtcp_f)
rtp_w.writerow(["arrival_ns", "wall_ns", "seq", "rtp_ts", "marker", "nal", "start", "size"])
rtcp_w.writerow(["arrival_ns", "wall_ns", "ntp_s", "rtp_ts", "pkt_count"])

t0 = time.monotonic_ns()
last_keepalive = time.monotonic()
sock.settimeout(10)
while True:
    now = time.monotonic_ns()
    if (now - t0) / 1e9 >= seconds:
        break
    if time.monotonic() - last_keepalive > 20:
        cseq += 1
        sock.sendall(f"GET_PARAMETER {base} RTSP/1.0\r\nCSeq: {cseq}\r\nSession: {session}\r\n\r\n".encode())
        last_keepalive = time.monotonic()
    while len(buf) < 4:
        recv_more()
    if buf[0:1] != b"$":
        # interleaved RTSP response (keepalive reply)
        read_response()
        continue
    ch, n = buf[1], struct.unpack(">H", buf[2:4])[0]
    while len(buf) < 4 + n:
        recv_more()
    pkt, buf = buf[4:4 + n], buf[4 + n:]
    arrival = time.monotonic_ns() - t0
    wall = time.time_ns()
    if ch == 0:
        b0, b1, seq, ts = struct.unpack(">BBHI", pkt[:8])
        cc = b0 & 0x0F
        off = 12 + 4 * cc
        if b0 & 0x10:  # header extension
            ext_len = struct.unpack(">H", pkt[off + 2:off + 4])[0]
            off += 4 + 4 * ext_len
        payload = pkt[off:]
        if b0 & 0x20:  # padding
            payload = payload[:-payload[-1]]
        nal_hdr = payload[0]
        ntype = nal_hdr & 0x1F
        start = 1
        if ntype == 28:  # FU-A
            fu = payload[1]
            start = int(bool(fu & 0x80))
            real = (nal_hdr & 0xE0) | (fu & 0x1F)
            if start:
                h264.write(b"\x00\x00\x00\x01" + bytes([real]))
            h264.write(payload[2:])
            ntype = fu & 0x1F
        elif ntype == 24:  # STAP-A
            i = 1
            types = []
            while i + 2 <= len(payload):
                ln = struct.unpack(">H", payload[i:i + 2])[0]
                nal = payload[i + 2:i + 2 + ln]
                h264.write(b"\x00\x00\x00\x01" + nal)
                types.append(str(nal[0] & 0x1F))
                i += 2 + ln
            ntype = "stap:" + "+".join(types)
        else:
            h264.write(b"\x00\x00\x00\x01" + payload)
        rtp_w.writerow([arrival, wall, seq, ts, int(bool(b1 & 0x80)), ntype, start, len(payload)])
    elif ch == 1:
        i = 0
        while i + 4 <= len(pkt):
            pt = pkt[i + 1]
            ln = (struct.unpack(">H", pkt[i + 2:i + 4])[0] + 1) * 4
            if pt == 200:  # Sender Report
                ntp_hi, ntp_lo, rtp_ts, pcount = struct.unpack(">IIII", pkt[i + 8:i + 24])
                rtcp_w.writerow([arrival, wall, ntp_hi - 2208988800 + ntp_lo / 2**32, rtp_ts, pcount])
            i += ln

try:
    cseq += 1
    sock.sendall(f"TEARDOWN {base} RTSP/1.0\r\nCSeq: {cseq}\r\nSession: {session}\r\n\r\n".encode())
except OSError:
    pass
sock.close()
for f in (h264, rtp_f, rtcp_f):
    f.close()
print(f"{name}: done", flush=True)
