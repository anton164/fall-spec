import time

def flatten(t):
    return [item for sublist in t for item in sublist]

class Timer(object):
    def __init__(self, message: str):
        self.message = message
      
    def __enter__(self):
        self.start_time = time.time()
        
  
    def __exit__(self, etype, value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"[Timer]: {self.message} took {elapsed_time}s")