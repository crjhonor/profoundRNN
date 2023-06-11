import time
from tqdm import tqdm

print("begin")
for i in tqdm(range(1000), ncols=100, desc="OPEN THE DOOR", colour="blue"):
    time.sleep(0.001)
print("DONE!")

print("begin again.")
for i in tqdm(range(1000), ncols=100, desc="GET it", colour = "magenta"):
    time.sleep(0.001)
print("DONE!")

print("begin third time.")
for i in tqdm(range(1000), ncols=100, desc="RIGHT", colour = "red"):
    time.sleep(0.001)
print("DONE!")