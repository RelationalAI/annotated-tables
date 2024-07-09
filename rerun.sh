command="python rel_dataset/kaggle_synth.py"

while ! $command; do
    echo "Command failed. Retrying..."
    sleep 1  # Optional: Wait for 1 second before retrying
done

echo "Command succeeded."
