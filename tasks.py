from invoke import Context
from invoke import task


@task
def fmt(c: Context):
    c.run("ruff check --fix-only --exit-zero", echo=True)
    c.run("ruff format", echo=True)
