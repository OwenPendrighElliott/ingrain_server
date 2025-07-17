import ingrain
import time
import threading
import statistics

# Configuration
model_name = "intfloat/e5-small-v2"
# model_name="Qwen/Qwen3-Embedding-0.6B"
# model_name = "intfloat/e5-base-v2"
# model_name = "Snowflake/snowflake-arctic-embed-m"
num_threads = 100
num_requests_per_thread = 100
delay_between_requests = 0

# Thread-safe structure to store response times
response_times = []
inference_times = []
response_times_lock = threading.Lock()

client = ingrain.Client()
client.load_sentence_transformer_model(name=model_name)


# Benchmarking function
def benchmark(thread_id):
    # Create a client and load the model for each thread
    client = ingrain.Client()
    # model = client.load_sentence_transformer_model(name=model_name)

    for _ in range(num_requests_per_thread):
        start_time = time.perf_counter()

        response = client.infer(name=model_name, text="Hello, world!")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Store inference time and response time
        inference_times.append(response["processingTimeMs"])

        with response_times_lock:
            response_times.append(elapsed_time)

        # Delay between requests to avoid spikes
        time.sleep(delay_between_requests)


# Measure the total benchmarking time
start_benchmark_time = time.perf_counter()

# Run benchmark
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=benchmark, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

end_benchmark_time = time.perf_counter()
total_benchmark_time = end_benchmark_time - start_benchmark_time

# Calculate QPS
total_requests = len(response_times)
qps = total_requests / total_benchmark_time

# Calculate statistics
mean_response_time = statistics.mean(response_times)
median_response_time = statistics.median(response_times)
stddev_response_time = statistics.stdev(response_times)

mean_inference_time = statistics.mean(inference_times)
median_inference_time = statistics.median(inference_times)
stddev_inference_time = statistics.stdev(inference_times)

print("Benchmark results:")
print(f"Concurrent threads: {num_threads}")
print(f"Requests per thread: {num_requests_per_thread}")
print(f"Total requests: {total_requests}")
print(f"Total benchmark time: {total_benchmark_time:.2f} seconds")
print(f"QPS: {qps:.2f}")
print(f"Mean response time: {mean_response_time:.4f} seconds")
print(f"Median response time: {median_response_time:.4f} seconds")
print(f"Standard deviation of response times: {stddev_response_time:.4f} seconds")
print(f"Mean inference time: {mean_inference_time:.4f} ms")
print(f"Median inference time: {median_inference_time:.4f} ms")
print(f"Standard deviation of inference times: {stddev_inference_time:.4f} ms")
