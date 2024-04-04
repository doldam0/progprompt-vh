import argparse


class RunEvalArguments(argparse.Namespace):
    progprompt_path: str
    expt_name: str
    openai_api_key: str
    unity_filename: str
    port: str
    display: str
    gpt_version: str
    env_id: int
    test_set: str
    prompt_task_examples: str
    seed: int
    prompt_num_examples: int
    prompt_task_examples_ablation: str
    load_generated_plans: bool


def parse_args() -> RunEvalArguments:
    parser = argparse.ArgumentParser()

    parser.add_argument("--progprompt-path", type=str, required=True)
    parser.add_argument("--expt-name", type=str, required=True)

    parser.add_argument("--openai-api-key", type=str, default="sk-xyz")
    parser.add_argument(
        "--unity-filename", type=str, default="/path/to/macos_exec.v2.3.0.app"
    )
    parser.add_argument("--port", type=str, default="8000")
    parser.add_argument("--display", type=str, default="0")

    parser.add_argument(
        "--gpt-version",
        type=str,
        default="text-davinci-002",
        choices=["text-davinci-002", "davinci", "code-davinci-002"],
    )
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument(
        "--test-set",
        type=str,
        default="test_unseen",
        choices=[
            "test_unseen",
            "test_seen",
            "test_unseen_ambiguous",
            "env1",
            "env2",
        ],
    )

    parser.add_argument(
        "--prompt-task-examples",
        type=str,
        default="default",
        choices=["default", "random"],
    )
    # for random task examples, choose seed
    parser.add_argument("--seed", type=int, default=0)

    ## NOTE: davinci or older GPT3 versions have a lower token length limit
    ## check token length limit for models to set prompt size:
    ## https://platform.openai.com/docs/models
    parser.add_argument(
        "--prompt-num-examples", type=int, default=3, choices=range(1, 7)
    )
    parser.add_argument(
        "--prompt-task-examples-ablation",
        type=str,
        default="none",
        choices=["none", "no_comments", "no_feedback", "no_comments_feedback"],
    )

    parser.add_argument("--load-generated-plans", type=bool, default=False)

    args = parser.parse_args(namespace=RunEvalArguments())
    return args
