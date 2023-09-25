import sky

from llm_atc.launch import train_task


def test_train():
    task = train_task(
        "vicuna",
        checkpoint_bucket="llm-atc",
        checkpoint_path="myvicuna",
        checkpoint_store="S3",
        finetune_data="./vicuna_test.json",
        name="myvicuna",
        cloud="aws",
        region="us-east-2",
        accelerator="A100:8",
        envs="HF_TOKEN=huggingFaceToken",
    )
    assert task.name == "myvicuna"
    assert task.envs["MODEL_NAME"] == "myvicuna"
    assert task.envs["HF_TOKEN"] == "huggingFaceToken"

    sky.launch(task, cluster_name="dummycluster", dryrun=True)
