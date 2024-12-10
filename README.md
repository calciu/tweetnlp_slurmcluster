# tweetnlp_slurmcluster
A transformer based model (RoBERTa) is used that was fine-tuned on tweeter data and specialized in sentiment-analysis from the tweetnlp module. It is a well performing big model with a rather manageable size (2-3 Gb in memory).
To achieve both node and core parallelism, we use the Ray module's hierarchical task scheduling.
In the SPLIT phase the texts (tweets) array is split into a number of chunks that equals the available nodes (machines) in the cluster and each chunk is divided in sub-chunks according to the available cores on one machine. The node chunks and core sub-chunks are kept in lists of split text arrays.
The MAP phase applies the annotation function model.sentiment(tweet) returning prediction or inference, on each sub-chunk of a chunk, using the so called list comprehension, to process simultaneously sub-chunks on the mobilised cores, which is then accordingly included at node level to process chunks in parallel on all nodes.
The function that processes core subchunks which contains the big model, needs to be wrapped successively into @ray.remote processes at core and node level that get it from shared memory where it had been previously stored (put) as a large object. 

# Steps

1. Take a big csv file containing at least two columns ("id" and "text") clean the text column with the script tweetsClean.py

2. The resulting file tw_out.csv that contains only the two above mentioned columns will be used in roberta_ray_slurm.py python script which is called by ray_roberta.slurm

3. On the cluster execute : sbatch ray_roberta.slurm
