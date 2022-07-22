import os

print("Test")

if os.path.exists("config.yaml"):
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)


