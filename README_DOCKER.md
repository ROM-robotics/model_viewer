## အသင့်သုံး docker image

### pull and run
```bash
docker pull romrobotics/model_viewer:latest
docker run --rm -it --network host romrobotics/model_viewer:latest
```

`--network host` is required: the Detailed (Netron) view starts a second
local server on a random port per uploaded model, so the container needs to
share the host's network namespace instead of relying on `-p` port mapping
(which can't predict that random port ahead of time). Once Flask starts, the
container's terminal output shows the address to open, e.g.:

```
 * Running on http://127.0.0.1:5000
```