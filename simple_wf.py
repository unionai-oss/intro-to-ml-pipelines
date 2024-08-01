# Import libraries and modules
# task
from flytekit import task, workflow

@task
def hello_world(name: str) -> str:
    return f"Hello {name}"

# workflow
@workflow
def main(name: str) -> str:
    return hello_world(name=name)