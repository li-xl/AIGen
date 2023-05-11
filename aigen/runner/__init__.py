from . import runner_3d

from gen.utils.registry import RUNNERS, build_from_cfg
from gen.utils.config import get_cfg

def build_runner():
    cfg = get_cfg()
    runner_cfg = cfg.runner 
    if runner_cfg is None or "type" not in runner_cfg:
        print("runner is not defined in config")
        exit(0)

    runner = build_from_cfg(runner_cfg,RUNNERS)
    return runner