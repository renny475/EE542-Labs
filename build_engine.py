import sys, time
import tensorrt as trt

onnx_path, engine_path = sys.argv[1], sys.argv[2]

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()          # TensorRT 11: always strongly typed
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(onnx_path):
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

# optimization profile: batch 1..32, tuned for 8
profile = builder.create_optimization_profile()
profile.set_shape("input", min=(1, 3, 224, 224), opt=(8, 3, 224, 224), max=(32, 3, 224, 224))
config.add_optimization_profile(profile)

t0 = time.time()
engine = builder.build_serialized_network(network, config)
if engine is None:
    sys.exit("build failed")
open(engine_path, "wb").write(engine)
print(f"wrote {engine_path} ({engine.nbytes/1e6:.1f} MB) in {time.time()-t0:.1f}s")
