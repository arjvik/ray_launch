# ray_launch

single-file [ray](https://www.ray.io/) helper, to "stop fighting ray and instead make ray fight for us"

uses ray for only what it's good at—distributing tasks across nodes—not for what it tries to be good for—enterprisey ml training tools

comment out decorators to debug on head node

#### usage

```python
import socket

@ray_launch.distribute
def main(args, rank=0, world_size=1):
    sleep(5)
    print(f"{socket.gethostname()} - {args=} {rank=} {world_size=}")
    
main("test")

# --------------------

from ray.experimental.tqdm_ray import tqdm

@ray_launch.parallelize
def process(tasks, a, b):
    print(f"{socket.gethostname()} - {tasks=} {a=} {b=}")
    for task in tqdm(tasks):
        pass

process(list(range(10)), a=1, b=2)

# --------------------

import torch

@ray_launch.distribute
def torch_main(args, rank=0, world_size=1):
    ray_launch.torch_init_process_group(rank, world_size)

    print('Awaiting all processes...')
    torch.distributed.barrier()

torch_main({"a":1, "b":2})
```
