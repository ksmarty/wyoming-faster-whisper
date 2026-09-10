# wyoming-faster-whisper (GPU fork)

A fork of [OHF-Voice/wyoming-faster-whisper](https://github.com/OHF-Voice/wyoming-faster-whisper)
that is synced with upstream and publishes a prebuilt NVIDIA GPU Docker image
to GitHub Container Registry. Everything else — models, options, environment
variables, Home Assistant name biasing — is documented in the
[original README](https://github.com/OHF-Voice/wyoming-faster-whisper#readme).

## Run the GPU version

Pull the prebuilt image (built from `Dockerfile.gpu` by the
"Build and Push GPU Image" workflow):

```sh
docker pull ghcr.io/ksmarty/wyoming-whisper-gpu:latest
```

Run it — the host needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed:

```sh
docker run -it --gpus all -p 10300:10300 -v /path/to/local/data:/data \
    ghcr.io/ksmarty/wyoming-whisper-gpu:latest --language en
```

The image defaults to `--device cuda` (float16 compute) and the
`Systran/faster-whisper-small` model. Pass options on the command line or via
`WYO_WHISPER_*` environment variables (see the
[original README](https://github.com/OHF-Voice/wyoming-faster-whisper#readme)).

## Docker Compose

The environment variables are the natural way to configure the container in a
stack, and the device reservation below is what hands it the GPU:

```yaml
services:
  whisper-gpu:
    image: ghcr.io/ksmarty/wyoming-whisper-gpu:latest
    container_name: whisper-gpu
    restart: unless-stopped
    environment:
      WYO_WHISPER_DEVICE: cuda:1    # defaults to cuda; :1 picks the second GPU
      WYO_WHISPER_MODEL: Systran/faster-distil-whisper-medium.en
      WYO_WHISPER_LANGUAGE: en
    ports:
      - "10301:10300"
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: ["gpu", "utility", "compute"]
```

If your Compose or Swarm version ignores `deploy:`, replace the whole
`deploy:` block with a single `runtime: nvidia` line.

Verify the container actually got the GPU:

```sh
docker inspect whisper-gpu --format '{{json .HostConfig.DeviceRequests}}'
docker exec whisper-gpu nvidia-smi
```

### Faster-whisper and float16

faster-whisper asks for `float16` compute by default on CUDA. If the GPU is not
actually reachable the container fails at startup with:

```
ValueError: Requested float16 compute type, but the target device or backend
do not support efficient float16 computation.
```

If your GPU cannot use float16, or you keep hitting that error while debugging
the GPU setup, run the `qwen3-asr` backend instead — an onnxruntime model that
uses its own compute type:

```yaml
    environment:
      WYO_WHISPER_DEVICE: cuda:1
      WYO_WHISPER_MODEL: andrewleech/qwen3-asr-0.6b-onnx
      WYO_WHISPER_LANGUAGE: en
      WYO_WHISPER_STT_LIBRARY: qwen3-asr
```

### Home Assistant

Set up the Wyoming STT integration and point it at the published port —
`tcp://<host>:10301` in the example above. Use `WYO_WHISPER_HASS_TOKEN` and
`WYO_WHISPER_HASS_API` to feed entity names into the prompt (see *Biasing
Toward Your Home Assistant Names* in the
[original README](https://github.com/OHF-Voice/wyoming-faster-whisper#readme)).