# import requests
# import time
# import threading

# # Configuration
# model_name = "intfloat/e5-small-v2"
# num_threads = 50
# num_requests_per_thread = 100

# # Load the model
# response = requests.post(
#     "http://localhost:8686/load_sentence_transformer_model",
#     json={"model_name": model_name},
# )
# print(response.json())

# # Benchmarking function
# def benchmark(thread_id):
#     start_time = time.time()
#     for _ in range(num_requests_per_thread):
#         response = requests.post(
#             "http://localhost:8686/infer_text",
#             json={"model_name": model_name, "text": "a cat"}
#         )
#         if response.status_code != 200:
#             print(f"Thread {thread_id} received an error: {response.json()}")
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Thread {thread_id} finished in {elapsed_time:.2f} seconds")

# # Run benchmark
# threads = []
# for i in range(num_threads):
#     thread = threading.Thread(target=benchmark, args=(i,))
#     threads.append(thread)
#     thread.start()

# for thread in threads:
#     thread.join()

# # Calculate QPS
# total_requests = num_threads * num_requests_per_thread
# total_time = sum(thread.join() for thread in threads)
# qps = total_requests / total_time

# print(f"Total requests: {total_requests}")
# print(f"Total time: {total_time:.2f} seconds")
# print(f"QPS: {qps:.2f}")

import requests
import time
import threading
import statistics

# Configuration
model_name = "intfloat/e5-small-v2"
num_threads = 10
num_requests_per_thread = 10
delay_between_requests = 0.01  # 10ms delay between requests

# Load the model
response = requests.post(
    "http://localhost:8686/load_sentence_transformer_model",
    json={"model_name": model_name},
)
print(response.json())

# Thread-safe structure to store response times
response_times = []
inference_times = []
response_times_lock = threading.Lock()

# Benchmarking function
def benchmark(thread_id):
    for _ in range(num_requests_per_thread):
        start_time = time.perf_counter()
        response = requests.post(
            "http://localhost:8686/infer_text",
            json={"model_name": model_name, "text": "a cat"}
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        data = response.json()
        inference_times.append(data["processingTimeMs"])

        # Store the response time
        with response_times_lock:
            response_times.append(elapsed_time)

        if response.status_code != 200:
            print(f"Thread {thread_id} received an error: {response.json()}")

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

print(f"Total requests: {total_requests}")
print(f"Total benchmark time: {total_benchmark_time:.2f} seconds")
print(f"QPS: {qps:.2f}")
print(f"Mean response time: {mean_response_time:.4f} seconds")
print(f"Median response time: {median_response_time:.4f} seconds")
print(f"Standard deviation of response times: {stddev_response_time:.4f} seconds")
print(f"Mean inference time: {mean_inference_time:.4f} ms")
print(f"Median inference time: {median_inference_time:.4f} ms")
print(f"Standard deviation of inference times: {stddev_inference_time:.4f} ms")

