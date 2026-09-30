SEED = 42

BASE_MODEL = "sapienzanlp/Minerva-7B-instruct-v1.0"

# outputs of each stage (each stage can be re-run from its files)
BONITA_DIR = "bonita_full"
WIKI_PATH = "wiki_passages.jsonl"
GENERATED_PATH = "generated_tasks.jsonl"
GENERATION_STATS_PATH = "generation_stats.json"
STUDENT_DIR = "student_bonita"
RESULTS_PATH = "extrinsic_results.json"
PREDICTIONS_PATH = "extrinsic_predictions.jsonl"

TEST_FRAC = 0.2  # share of BoolQ held out as the final test set of the base model and the student
N_PASSAGES = 10_000  # unannotated paragraphs given to BONITA (~ size of BoolQ train)

# one of the 10 files of Italian Wikipedia (~180k articles) is enough to sample the passages
WIKI_SHARD = "hf://datasets/wikimedia/wikipedia/20231101.it/train-00000-of-00010.parquet"

# for both, the checkpoint with the lowest loss on a held-out 5% of the training data is kept
BONITA_EPOCHS = 4
STUDENT_EPOCHS = 4
DEV_FRAC = 0.05
