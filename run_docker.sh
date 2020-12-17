echo killing old docker containers
docker container stop mask_validator_container
docker container rm mask_validator_container
echo building docker image
docker build -t mask_validator_image ./fastapi_app
echo running docker container
docker run -d --name mask_validator_container -p 8080:8080 -e PORT="8080" mask_validator_image