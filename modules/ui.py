import os
import importlib
from typing import *

import gradio as gr
import gradio.routes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Tab:
    TABS_DIR = os.path.join(ROOT_DIR, "modules", "tabs")

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def sort(self):
        return 1

    def title(self):
        return ""

    def ui(self, outlet: Callable):
        pass

    def __call__(self):
        children_dir = self.filepath[:-3]
        children = []

        if os.path.isdir(children_dir):
            for file in os.listdir(children_dir):
                if not file.endswith(".py"):
                    continue
                module_name = file[:-3]
                parent = os.path.relpath(Tab.TABS_DIR, Tab.TABS_DIR).replace("/", ".")

                if parent.startswith("."):
                    parent = parent[1:]
                if parent.endswith("."):
                    parent = parent[:-1]

                children.append(
                    importlib.import_module(f"modules.tabs.{parent}.{module_name}")
                )

        children = sorted(children, key=lambda x: x.sort())

        tabs = []

        for child in children:
            attrs = child.__dict__
            tab = [x for x in attrs.values() if issubclass(x, Tab)]
            if len(tab) > 0:
                tabs.append(tab[0])

        def outlet():
            with gr.Tabs():
                for tab in tabs:
                    with gr.Tab(tab.title()):
                        tab()

        return self.ui(outlet)


def load_tabs() -> List[Tab]:
    tabs = []
    files = os.listdir(os.path.join(ROOT_DIR, "modules", "tabs"))

    for file in files:
        if not file.endswith(".py"):
            continue
        module_name = file[:-3]
        module = importlib.import_module(f"modules.tabs.{module_name}")
        attrs = module.__dict__
        TabClass = [
            x
            for x in attrs.values()
            if type(x) == type and issubclass(x, Tab) and not x == Tab
        ]
        if len(TabClass) > 0:
            tabs.append((file, TabClass[0]))

    tabs = sorted([TabClass(file) for file, TabClass in tabs], key=lambda x: x.sort())
    return tabs


def create_ui():
    block = gr.Blocks()

    with block:
        with gr.Tabs():
            tabs = load_tabs()
            for tab in tabs:
                with gr.Tab(tab.title()):
                    tab()

    return block