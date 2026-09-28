import socket
import struct
import threading

import msgpack

from hubclient import HubConnection


def test_hello_send_recv():
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    got = {}

    def serve():
        conn, _ = server.accept()
        f = conn.makefile("rb")
        for key in ("hello", "result"):
            (n,) = struct.unpack(">I", f.read(4))
            got[key] = msgpack.unpackb(f.read(n), raw=False)
        body = msgpack.packb({"type": "frame", "data": b"\x00\x00\x00\x01e"}, use_bin_type=True)
        conn.sendall(struct.pack(">I", len(body)) + body)
        conn.close()

    t = threading.Thread(target=serve)
    t.start()
    with HubConnection(f"127.0.0.1:{port}", {"role": "cv", "id": "w"}) as c:
        c.send({"type": "result", "camera": "grey", "pts": 1, "dets": []})
        msg = c.recv()
    t.join()
    assert got["hello"] == {"role": "cv", "id": "w", "type": "hello"}
    assert got["result"]["camera"] == "grey"
    assert msg == {"type": "frame", "data": b"\x00\x00\x00\x01e"}
