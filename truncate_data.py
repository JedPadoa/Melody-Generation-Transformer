import json

FULL_DATA_PATH = "train.json"
TRUNCATED_DATA_PATH = "first_1000.json"

def get_first_1000():
    
    with open(FULL_DATA_PATH, 'r') as data:
        data = json.load(data)
        first_1000 = data[:1000]
    
    with open (TRUNCATED_DATA_PATH, 'w') as file:
        json.dump(first_1000, file)
        
    return first_1000
    
if __name__ == "__main__":
    get_first_1000()