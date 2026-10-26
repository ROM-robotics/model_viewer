import netron

# Netron releases the OS socket of a stopped server asynchronously, so
# immediately rebinding the same fixed port on the next upload can raise
# "Address already in use". Letting the OS pick a free port (port 0) avoids
# that race entirely; the previous server is asked to stop in the background.
_last_address = None


def start_netron(path: str) -> str:
    """Launch a local Netron server for the given model file and return the
    URL to embed for the detailed graph view."""
    global _last_address
    if _last_address is not None:
        try:
            netron.stop(_last_address)
        except Exception:
            pass
    host, port = netron.start(path, address=("127.0.0.1", 0), browse=False)
    _last_address = (host, port)
    return f"http://{host}:{port}"
