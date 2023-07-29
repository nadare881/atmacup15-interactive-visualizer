# argparseで引数呼んでuiを動かす

import os
import argparse
from modules import ui

parser = argparse.ArgumentParser()

parser.add_argument("--host", help="Host to connect to", type=str, default="127.0.0.1")  # localhost
parser.add_argument("--port", help="Port to connect to", type=int, default=8015)
parser.add_argument("--share", help="Enable gradio share", action="store_true")

cmdargs, _ = parser.parse_known_args()

def webui():
    app = ui.create_ui()
    app.queue(64)
    app, local_url, share_url = app.launch(
        server_name=cmdargs.host,
        server_port=cmdargs.port,
        share=cmdargs.share,
    )


if __name__ == "__main__":
    webui()