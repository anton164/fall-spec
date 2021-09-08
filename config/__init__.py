import json

try:
    with open("./config/twitter.json", "r") as f:
        twitter_keys = json.load(f)
        print("Successfully loaded Twitter API keys")
except:
    print("Falling back to loading Twitter API keys from sample file...")
    print("Have you created secrets/twitter.json with the correct API keys?")
    with open("./config/twitter_sample.json", "r") as f:
        twitter_keys = json.load(f)