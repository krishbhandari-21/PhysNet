"""PhysNet physics module."""
from .base import BasePINN
from .heat import HeatPINN
from .burgers import BurgersPINN
from .poisson import PoissonPINN

__all__ = ["BasePINN", "HeatPINN", "BurgersPINN", "PoissonPINN"]
