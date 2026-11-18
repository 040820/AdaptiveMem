import argparse
import logging
import os
from functools import partial, wraps
from pathlib import Path
import sys

from dotenv import load_dotenv

run_dir = Path(__file__).resolve().parent
repo = run_dir.parents[1]
sys.path.insert(0, str(repo))
load_dotenv(repo / ".env")
# Only the child reproduction process connects directly to the configured intranet services.
for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(proxy_var, None)
    os.environ.pop(proxy_var.lower(), None)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--preflight-only", action="store_true")
args, _ = parser.parse_known_args()
llm_url = os.environ["LICO_LLM_BASE_URL"]
llm_model = os.environ["LICO_LLM_MODEL"]
llm_key = os.environ["LICO_LLM_API_KEY"]
embedding_url = os.environ["LICO_EMBEDDING_BASE_URL"]
embedding_model = os.environ["LICO_EMBEDDING_MODEL"]
embedding_key = os.environ["LICO_EMBEDDING_API_KEY"]
embedding_dim = int(os.environ["LICO_EMBEDDING_DIMENSIONS"])
assert ":8001/v1" in llm_url and ":8015/v1" in embedding_url
assert llm_model == "qwen3-30b-a3b" and embedding_model == "bge-m3"

os.chdir(run_dir)
sys.argv = [sys.argv[0], "--data", "locomo", "--sample", "26", "--model", "qwen", "--file", "vllm"]
from common import config

config.datapath = str(repo / "data" / "dataset_locomo.json")
config.API_KEY = llm_key
config.OPENROUTER_URL = llm_url
config.MODEL = config.RE_MODEL = config.MODEL_SORT = config.CHOOSE_MODEL = llm_model

from llm import controller, embeddings
embeddings.EMBED_API_KEY = embedding_key
embeddings.EMBED_BASE_URL = embedding_url
os.environ["OPENAI_API_KEY"] = embedding_key

from llm import rag_utils
rag_utils.get_openai_embedding = partial(embeddings.get_openai_embedding, model=embedding_model)

base_chat = controller.LLM.chat_with_tool

@wraps(base_chat)
def local_chat(self, **kwargs):
    kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
    return base_chat(self, **kwargs)

controller.LLM.chat_with_tool = local_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if args.preflight_only:
    vec = rag_utils.get_embeddings(["LoCoMo preflight"])
    assert vec.shape == (1, embedding_dim), vec.shape
    reply = controller.LLM().chat_with_tool(
        messages=[{"role": "user", "content": "Reply with OK."}], use_tool=False
    )
    assert reply.choices[0].message.content.strip(), reply
    print(f"Preflight OK: {llm_model}, {embedding_model}, {vec.shape[1]} dimensions")
else:
    import run
    run.main()
