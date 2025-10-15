#!/bin/bash
# Monitor training progress

echo "Monitoring fine-tuning progress..."
echo "Log file: /workspaces/AI4SE_assignment1/finetune_output.log"
echo ""

while true; do
    if [ -f "/workspaces/AI4SE_assignment1/finetune.pid" ]; then
        PID=$(cat /workspaces/AI4SE_assignment1/finetune.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "$(date '+%H:%M:%S') - Fine-tuning running (PID: $PID)"
            echo "Last 5 lines of log:"
            tail -5 /workspaces/AI4SE_assignment1/finetune_output.log 2>/dev/null | sed 's/^/  /'
            echo ""
        else
            echo "$(date '+%H:%M:%S') - Fine-tuning completed!"
            echo "Final log:"
            tail -20 /workspaces/AI4SE_assignment1/finetune_output.log 2>/dev/null | sed 's/^/  /'
            break
        fi
    else
        echo "PID file not found"
        break
    fi
    sleep 30
done
