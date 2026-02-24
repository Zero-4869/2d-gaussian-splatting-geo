import os
import itertools
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outname', type=str)
parser.add_argument('--use_gt', action='store_true', default=False)
parser.add_argument('--use_geo', action='store_true', default=False)
args = parser.parse_args()

scenes = [ # "scene_01",
            # "scene_02",
           # "scene_03",
          #  "scene_04",
           # "scene_05",
           # "scene_06",
           "scene_07",
           # "scene_08",
]

factors = [2]

output_dir = "output/translab"

dataset_dir = '/home/hongyuzhou/Datasets/translab'

out_name = args.outname

excluded_gpus = set([])

jobs = list(itertools.product(scenes, factors))

def train_scene(gpu, scene, factor):

    cmd = f'rm -rf {output_dir}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = ""
    common_args += " --use_gt" if args.use_gt else ""
    common_args += " --use_geo" if args.use_geo else ""
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -r2 -s {dataset_dir}/{scene} -m {output_dir}/{scene}/{out_name} --eval --depth_ratio 1.0 \
    #     --lambda_dist 1000 --port {6209+int(gpu)} --test_iterations 30000 --mask_background {common_args}"
    
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene} -m {output_dir}/{scene}/{out_name} --eval \
        --lambda_normal 0.0 --port {6209+int(gpu)} --test_iterations 30000 --mask_background {common_args}"
    
    print(cmd)
    os.system(cmd)

    common_args = ""
    common_args += " --use_geo" if args.use_geo else ""
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene}/{out_name} --skip_train --skip_mesh --mask_background {common_args}"
    print(cmd)
    os.system(cmd)

    common_args = ""
    common_args += " --use_geo" if args.use_geo else ""
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene}/{out_name} --skip_train --skip_test --mask_background --num_cluster 5 \
        --voxel_size 0.002 --depth_trunc 10.0 {common_args}"
    print(cmd)
    os.system(cmd)
        
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}/{out_name}"
    print(cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_translab/eval.py --data {output_dir}/{scene}/{out_name}/train/ours_30000/fuse_post.ply \
          --scan {scene} --mode mesh --dataset_dir {dataset_dir} --vis_out_dir {output_dir}/{scene}/{out_name}/train/ours_30000/ \
          --downsample_density 0.002"
    print(cmd)
    os.system(cmd)
    
    return True

def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)