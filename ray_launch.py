import os
import inspect
import ray
from functools import wraps
from itertools import zip_longest
from time import sleep

_ray_context = ray.init('localhost:6379', runtime_env={"working_dir": os.getcwd()})

def distribute(f):
    """
    Run the same function on every single node in a ray cluster, with rank and world_size correctly set
    Comment out @ray_launch.distribute and the function will run only on the head node
    
    Intended to run a main() function which sets up a torch.distributed environment with the rank and world_size

    Note: Must include kwargs `rank=0` and `world_size=1`!

    Usage:
    ```
    import socket
    
    @ray_launch.distribute
    def main(args, rank=0, world_size=1):
        sleep(5)
        print(f"{socket.gethostname()} - {args=} {rank=} {world_size=}")
        
    main("test")
    ```
    """
    assert all(a in inspect.getargs(f.__code__).args for a in ('rank', 'world_size')), "Include kwargs rank=0 and world_size=1!"
    @wraps(f)
    def wrapper(*args, **kwargs):
        gpu_count = int(ray.cluster_resources()['GPU'])
        @ray.remote(num_gpus=1)
        def run(r):
            sleep(1)
            return f(*args, **({**kwargs, 'rank': r, "world_size": gpu_count}))
        tasks = [run.remote(0)]
        sleep(0.5) # hope for rank 0 to land on head node
        tasks += [run.remote(i) for i in range(1, gpu_count)]
        return ray.get(tasks)
    return wrapper

def parallelize(f):
    """
    Run a function that processes a list of tasks by splitting up the task list onto each node, collecting the results transparently
    Comment out @ray_launch.parallelize and the tasks will only be processed on the head node

    Intended for data processing - you are encouraged to use ray.experimental.tqdm_ray.tqdm() for a progress bar.

    Usage:
    ```
    import socket
    
    @ray_launch.parallelize
    def process(tasks, a, b):
        print(f"{socket.gethostname()} - {tasks=} {a=} {b=}")

    process(list(range(10)), a=1, b=2)
    ```
    """
    @wraps(f)
    def wrapper(tasks, *args, **kwargs):
        @distribute
        def run(task, *args, rank, world_size, **kwargs):
            return f(task[rank::world_size], *args, **kwargs)
        results = run(tasks, *args, **kwargs)
        return [item for pair in zip_longest(*results)
                     for item in pair
                     if item is not None]
    return wrapper

def master_address():
    """
    Return the IP address of the master node, (hopefully) where the process with rank 0 is running
    """
    return _ray_context.address_info['address'].split(':')[0]

def node_address():
    """
    Return the IP address of the current node   
    """
    return _ray_context.address_info['node_ip_address']

def torch_init_process_group(rank, world_size, port=29500):
    """
    Initialize torch.distributed process group with one Ray task per GPU

    Intended to be called from a @distribute main() function, passing in the rank and world_size kwargs

    E.g.
    ```
    @ray_launch.distribute
    def main(args, rank=0, world_size=1):
        ray_launch.torch_init_process_group(rank, world_size)
    
    main(...)
    ```
    """
    if rank == 0:
        assert node_address() == master_address(), "Rank 0 got placed on node other than head node!"
    import torch
    torch.distributed.init_process_group('nccl', init_method=f'tcp://{master_address()}:{port}', rank=rank, world_size=world_size)
    torch.distributed.barrier()
