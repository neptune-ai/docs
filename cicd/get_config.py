#!/usr/bin/python3

import os
import sys

import yaml
from git import Repo
from jinja2 import Environment, FileSystemLoader


def create_values(env, registry, image):
    with open("dependencies.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        template_env = Environment(loader=FileSystemLoader(data["environment"][env]["templates"]))
        template = template_env.get_template("neptune-docs.yaml.j2")
        return template.render(image=image, registry=registry)


def main():
    env = sys.argv[1]
    registry = sys.argv[2]
    image = sys.argv[3]

    print(create_values(env, registry, image))

if __name__ == "__main__":
    main()
