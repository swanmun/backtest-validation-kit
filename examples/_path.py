"""Lets the examples import vkit without installing it. / 설치하지 않고도 예제가 vkit을 불러오게 한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
