docker network create brainstorm-network
docker build -f ./Dockerfile -t brainstorm-ml .
docker build -f ./Dockerfile -t brainstorm-service .
docker run -d -p 8000:8000 --network=brainstorm-network -e SERVICE_URL="http://brainstorm-service-container:8081/api/task/updateStatus" -v "/tmp/docker:/tmp/docker" --name brainstorm-ml-container brainstorm-ml:latest
docker run -d -p 8081:8081 --network=brainstorm-network -v "/tmp/docker:/tmp/docker" --name brainstorm-service-container brainstorm-service:latest