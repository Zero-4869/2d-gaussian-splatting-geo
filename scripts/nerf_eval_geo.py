# training scripts for the nerf-synthetic datasets
# this script is adopted from GOF
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_nerf_synthetic.py
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools

scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
# scenes = ["drums"]

factors = [1]

output_dir = "output/exp_nerf_synthetic"

dataset_dir = '/home/hongyuzhou/Datasets/nerf_synthetic'

dry_run = False

excluded_gpus = set([])

out_name = "geo_depthgt_nogeo"


jobs = list(itertools.product(scenes, factors))

def train_scene(gpu, scene, factor):

    cmd = f'rm -rf {output_dir}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene} -m {output_dir}/{scene}/{out_name} --eval --depth_ratio 1.0 \
    #     --lambda_dist 1000 --port {6209+int(gpu)} --save_iterations 7000 16000 30000 --test_iterations 30000 --use_geo --white_background"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene} -m {output_dir}/{scene}/{out_name} --eval \
        --lambda_normal 0.0 --port {6209+int(gpu)} --save_iterations 7000 16000 30000 --test_iterations 30000 --use_geo --white_background --use_gt"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene}/{out_name} --skip_train --skip_mesh --use_geo"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene}/{out_name} --skip_train --skip_test --use_geo"
    print(cmd)
    if not dry_run:
        os.system(cmd)
        
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}/{out_name}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"python scripts/eval_nerf/eval.py --data {output_dir}/{scene}/{out_name}/train/ours_30000/fuse_post.ply \
          --scan {scene} --mode mesh --dataset_dir {dataset_dir} --vis_out_dir {output_dir}/{scene}/{out_name}/train/ours_30000/ \
          --downsample_density 0.002"
    print(cmd)
    if not dry_run:
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