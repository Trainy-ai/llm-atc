import sky

from llm_atc.launch import train_task


def test_train():
    task = train_task(
        "vicuna",
        finetune_data="./vicuna_test.json",
        name="myvicuna",
        cloud="gcp",
        accelerator="A100-80GB:4",
        envs="HF_TOKEN=huggingFaceToken",
    )
    assert task.name == "myvicuna"
    assert task.envs["MODEL_NAME"] == "myvicuna"
    assert task.envs["HF_TOKEN"] == "huggingFaceToken"

    sky.launch(task, cluster_name="dummycluster", dryrun=True)
