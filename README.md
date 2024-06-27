# Using huggingface model to get suggestions.

-- It will use docker container to run hugging-face model with fastapi to get suggestions or best options available from the list options.


Use following to build:-
`docker build -t local-fastapi .`

Use following to run:-
`docker run -p 5001:5001 local-fastapi`