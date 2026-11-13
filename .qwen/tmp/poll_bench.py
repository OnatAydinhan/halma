import time, os
f = "bench_depths_out.txt"
deadline = time.time() + 55
while time.time() < deadline and not os.path.exists(f):
    time.sleep(2)
print("DONE" if os.path.exists(f) else "still running")
