"""Set VIVIDRGB_RESEARCH_ROOT to the root of the extracted research assets."""
import os
from pathlib import Path

def research_path(relative=''):
    return Path(os.environ.get('VIVIDRGB_RESEARCH_ROOT', 'research-workspace')).resolve() / relative
