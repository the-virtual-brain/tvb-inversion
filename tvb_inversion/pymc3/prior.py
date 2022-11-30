from dataclasses import dataclass
from typing import List, Optional


class Prior:
    def __init__(
            self,
            param: Optional[List[str]] = None,
            loc: Optional[List[float]] = None,
            scale: Optional[List[float]] = None,
            dist: Optional[List[str]] = None
    ):
        if param is None:
            self.param = []
        else:
            self.param = param
        if loc is None:
            self.loc = []
        else:
            self.loc = loc
        if scale is None:
            self.scale = []
        else:
            self.scale = scale
        if dist is None:
            self.dist = []
        else:
            self.dist = dist

        self._update_attrs()

    def _update_attrs(self):
        for p, l, s, d in zip(self.param, self.loc, self.scale, self.dist):
            exec(f"self.{p} = {dict(loc=l, scale=s, dist=d)}", {"self": self})

    def append(self, param: str, loc: float, scale: float, dist: str):
        self.param.append(param)
        self.loc.append(loc)
        self.scale.append(scale)
        self.dist.append(dist)

        self._update_attrs()

