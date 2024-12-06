import os
import sys
import torch
import pandas as pd # to read csv file
import numpy as np  # to use np.vstack to combine results from jobs
import ray # to replace multiprocessing
import tweetnlp
import time

# Assuming tweets are stored in CSV format
tweets_df = pd.read_csv('tw_out.csv',delimiter=",", low_memory=False)
#tweets_df = pd.read_csv('tw100_out.csv',delimiter=",", low_memory=False)
tweets_df = tweets_df[0:1000]

# Get number of nodes and cores
cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
sarrcnt = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
chunksize = len(tweets_df) // sarrcnt

model = tweetnlp.load_model('sentiment')

def sentbert(tweet):
    return model.sentiment(tweet)
    #return len(tweet)


# Define function to process a subchunk of tweets
def process_subchunk(tweets_chunk):
    embeddings = []
    for tweet in tweets_chunk:
      outputs = sentbert(tweet)
      embeddings.append(outputs)
    return embeddings

fcorelev_ref = ray.put(process_subchunk)

# Define a Ray task for core level processing by wrapping the process_subchunk function
@ray.remote
def process_core_level(process_subchunk, subchunk):
    return process_subchunk(subchunk)

# Define function to process a chunk of tweets at node level
@ray.remote
def parallel_process_tweets(tweets_chunk, n_processes=8):
    # Split tweets chunk into smaller sub-chunks for each CPU core
    chunk_size = len(tweets_chunk) // n_processes # must be > 1 if not range error
    sub_chunks = [tweets_chunk[i:i + chunk_size] for i in range(0, len(tweets_chunk), chunk_size)]
    core_futures = [process_core_level.remote(fcorelev_ref,chunk) for chunk in sub_chunks] # use remote wrapped core level function
    # Collect and combine results
    results = ray.get(core_futures)

    # Flatten the results
    return [embedding for sublist in results for embedding in sublist]

# Function to load and chunk tweets for SLURM task
def load_and_chunk_tweets(task_id, chunk_size=1000):
    # Split data into chunks for parallel processing
    start_idx = task_id * chunk_size
    end_idx = start_idx + chunk_size
    return tweets_df["text"].iloc[start_idx:end_idx].tolist()



def main():
  st = time.time()
  # Initialize Ray with SLURM integration
  #ray.init(address='auto')
  # Start Ray cluster
  ray.init(num_cpus=cpus_per_task, ignore_reinit_error=True)
  # Load BERT model and tokenizer from NLP

  task_id = int(sys.argv[1]) # get the task identifier (from 0 to max in task array) 
  # Split tweets into chunks for each node (task)
  chunks = load_and_chunk_tweets(task_id, chunksize)
  chunks = [chunks] # array of lists needed probably
  # Parallel processing with Ray
  # Collect results
#  node_futures = [predict.remote(fnodelev_ref, chunk, cpus_per_task) for chunk in chunks]
  node_futures = [parallel_process_tweets.remote(chunk, cpus_per_task) for chunk in chunks]

  # Gather all results
  results = ray.get(node_futures)


  # Combine embeddings
  embeddings = np.vstack(results)
  #ed = time.time()
  #print(ed - st)
  # Save the embeddings
  output_file = f"roberta_ray_task_{task_id}.npy"
  np.save(output_file, embeddings)
  ray.shutdown()

if __name__ == "__main__":
    main()
