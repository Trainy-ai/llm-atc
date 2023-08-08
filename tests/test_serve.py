import sky

from llm_atc.serve import serve_route


def test_serve():
    serve_task = serve_route(
        ["lmsys/vicuna-7b-1.3"],
        accelerator="V100:1",
        envs="HF_TOKEN=mytoken",
        cloud="aws",
        region="us-east-2",
    )
    assert serve_task.envs["MODELS_LIST"] == "lmsys/vicuna-7b-1.3"
    assert serve_task.envs["HF_TOKEN"] == "mytoken"

    sky.launch(serve_task, cluster_name="dummycluster", dryrun=True)
