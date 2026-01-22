from pyremote import remote, UvConfig

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123",
    uv=UvConfig(path="~/.venv", python_version="3.12"),
    timeout=60
)
def streaming_loop():
    import time
    import sys
    
    print(f"Python version: {sys.version}")
    print("Starting loop...")
    
    for i in range(5):
        print(f"Processing item {i+1}/5")
        time.sleep(1)
    
    print("Done!")
    return {"status": "completed", "items": 5}

if __name__ == "__main__":
    result = streaming_loop()
    print(f"Result: {result}")