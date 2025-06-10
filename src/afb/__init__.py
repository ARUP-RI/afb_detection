#!/usr/bin/env python3

import typer

# imported for logging configuration side effects
import afb.config  # noqa

import afb.inference
import afb.train

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)

app.add_typer(afb.inference.app, name="infer", help="Inference over data")
app.add_typer(afb.train.app, name="train", help="Train model")

if __name__ == "__main__":
    app()
