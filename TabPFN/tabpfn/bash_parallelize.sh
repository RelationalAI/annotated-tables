# Array to store PIDs of background processes
declare -a pids=()
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Function to clean up background processes
cleanup() {
    echo "Terminating background processes..."
    # Kill all background processes
    for pid in "${pids[@]}"; do
        kill "$pid"
    done
    exit 1
}

# Set trap to catch signals and run cleanup function
trap cleanup SIGINT

python run_kaggle_with_selected_indices.py 0 15000 & pids+=($!)
python run_kaggle_with_selected_indices.py 15000 30000 & pids+=($!)
python run_kaggle_with_selected_indices.py 30000 45000 & pids+=($!)
python run_kaggle_with_selected_indices.py 45000 60000 & pids+=($!)
python run_kaggle_with_selected_indices.py 60000 70000 & pids+=($!)


# Wait for all background processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All background processes have completed."