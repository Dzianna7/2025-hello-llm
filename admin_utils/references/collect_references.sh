# Run: nohup bash admin_utils/references/collect_references.sh > references.log 2>&1 &
# Monitor: ps -ef | grep [r]eferences
# Kill: kill -9 $(ps aux | grep '[r]eferences' | awk '{print $2}')

set -ex

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

source venv/bin/activate

export PYTHONPATH=$(pwd)

# GPU measurements are for 1x A5000 (24 GB)
# ~2 GPU min
python admin_utils/references/get_datasets_analytics.py

# ~1 GPU min
python admin_utils/references/get_model_analytics.py

# ~1 GPU min
python admin_utils/references/get_inference_analytics.py

# ~19 GPU min
python admin_utils/references/get_references.py

# ~9 GPU min
python admin_utils/references/get_sft_references.py

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

# Compare references
python admin_utils/references/comparison/comparator.py --old-references admin_utils/references/gold/reference_scores.json --new-references admin_utils/references/gold/reference_sft_scores.json

cat dist dist/compared_before_after_sft.csv
