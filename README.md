# ray_launch

single-file ray helper, to "stop fighting ray and instead make ray fight for us"

comment out decorators to debug on head node

#### usage

```python
import socket

@ray_launch.distribute
def main(args, rank=0, world_size=1):
    sleep(5)
    print(f"{socket.gethostname()} - {args=} {rank=} {world_size=}")
    
main("test")

@ray_launch.parallelize
def process(tasks, a, b):
    print(f"{socket.gethostname()} - {tasks=} {a=} {b=}")

process(list(range(10)), a=1, b=2)
```