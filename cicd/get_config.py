#!/usr/bin/python3

import os
import sys

import yaml
from git import Repo
from jinja2 import Environment, FileSystemLoader


if __name__ == "__main__":
    env = sys.argv[1]
    registry = sys.argv[2]
    image = sys.argv[3]

    template_directory = "cicd/templates/" + env

    template_env = Environment(loader=FileSystemLoader(template_directory))
    template = template_env.get_template("neptune-docs.yaml.j2")

    print(template.render(image=image, registry=registry))
