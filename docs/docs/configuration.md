# Configuration

There are environment variables which you can use to configure the behaviour of Ingrain. It is recommended to test different configurations to achieve optimal performance on your given hardware.

## Model Server

### `TRITON_GRPC_URL`

This tells the model server where to talk to Triton, it uses the GRPC interface for Triton so it must be the URL and port for GRPC.

### `MAX_LOADED_MODELS`

The is the maximum number of models which can be loaded at once. The default is 5 but you may wish to increase it if you have a need for more and also have sufficient RAM and VRAM.

When `MAX_LOADED_MODELS` is exceeded you will need to unload a model before a new one can be loaded.

### `MAX_BATCH_SIZE`

This is the maximum batch size for models, it will change the configuration generated for Triton. If you send ingrain a request with more items than `MAX_BATCH_SIZE` you will get an error. The default is `32`.

### `DYNAMIC_BATCHING`

Must be `true` or `false`. This toggles the dynamic batching behaviour for models which are created, dynamic batching lets Triton create batches from separate requests which arrive close to each other. The default is `true`.

### `INSTANCE_KIND`

This should be set to match your hardware. If running Triton in a CPU then use `KIND_CPU`, otherwise use `KIND_GPU`.

### `TENSORRT_ENABLED`

This toggles TensorRT acceleration. Ingrain uses ONNX conversions of the models, Triton has the ability to dynamically convert these with TensorRT. The conversion process is very slow but can deliver a small inference speedup - the speedup is going to be model and hardware dependent. The default is `false`, it is probably not worth the overhead unless you need to squeeze an extra few percent out of the models. If set, valid values are `true` or `false`.

### `FP16_ENABLED`

If using `TENSORRT_ENABLED=true` then you can also set `FP16_ENABLED`. This will make the TensorRT conversions use FP16 precision which should accelerate inference even further. The default is `false`. If set, valid values are `true` or `false`.

### `MODEL_INSTANCES`

This is the number of model instances to set in the config. Additional model instances can help with concurrency on GPU hardware. The default is `0` (not set). Setting this to a small number in the range of `2` to `4` can improve throughput on some hardware at the cost of additional memory usage (each instance is a new copy of the model in memory). Cannot be negative.

## Inference Server

### `TRITON_GRPC_URL`

This tells the model server where to talk to Triton, it uses the GRPC interface for Triton so it must be the URL and port for GRPC.

## Triton

It is recommended that you run triton with `--model-control-mode=explicit`, this lets ingrain control the model loading and unloading.

It is also recommended to change the memory allocator to `jemalloc` with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so`. This appears to be better at freeing memory when models are loaded and unloaded which works well with Ingrain's usage patterns.