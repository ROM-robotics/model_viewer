- docker image ကို run လိုက်တာနဲ့ model viewer ကို အလုပ်လုပ်စေချင်တယ်။
- save button ကို နှိပ်ရင်တော့ host မှာ လာပြီးတော့ permission 777 နဲ့ png ကို save ပေးရမယ်။
  (Decision: "Save as PNG" က browser-side canvas download ပဲ ဖြစ်ပြီး container/host
  filesystem ကို မထိပါ — user ရဲ့ browser download folder ကို အလိုအလျောက် သွားပါတယ်။)

## Build

```bash
./docker/build_script
```

Builds the image as `romrobotics/model_viewer:latest`, based on Ubuntu 22.04
with all Python deps installed system-wide (no venv inside the container).

## Run

```bash
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