### Basic Experimental Setup

# Data Generation

Generate 3 separate datasets corresponding to Epsilon Values and save, as opposed to doing while in the model. This is useful to present as downstream artifacts in paper, and also for reproducibility. 

# Model training

Submit jobs to cluster with models being fed 3 different datasets. Model scripts record loss values, which can be compared across the 3 and insights can be drawn. values and model parameters are saved
with each job, which makes use of GPUs to hopefully train very fast. 