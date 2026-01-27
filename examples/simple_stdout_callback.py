from pyremote import remote, UvConfig

# Collect logs in real-time using a callback
logs = []

def log_handler(line: str):
    """Callback that receives each line of stdout as it streams."""
    logs.append(line)
    # You could also send to logging, websocket, database, etc.
    # Example: logger.info(line)
    # Example: websocket.send(line)

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123",
    uv=UvConfig(path="~/.venv", python_version="3.12"),
    timeout=60,
    stdout_callback=log_handler,
)
def task_with_logging():
    import time
    import sys
    
    print(f"Python version: {sys.version}")
    print("Starting task...")
    
    for i in range(5):
        print(f"Processing step {i+1}/5")
        time.sleep(1)
    
    print("Task completed!")
    return {"status": "completed", "steps": 5}

if __name__ == "__main__":
    result = task_with_logging()
    
    print("\n--- Captured Logs ---")
    for log in logs:
        print(f"  > {log}")
    
    print(f"\nResult: {result}")
    print(f"Total log lines captured: {len(logs)}")
