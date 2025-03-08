# MountainMadness

To run this: 

docker build -t my-ai-app .     
docker run -it --rm my-ai-app /bin/bash  


To create a multi-platform image:

docker build --platform linux/amd64,linux/arm64 -t my-ai-app .
