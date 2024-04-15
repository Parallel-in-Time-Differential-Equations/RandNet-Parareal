from typing import TYPE_CHECKING
from typing import Callable,TypeVar, Any

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax import Array
    from jax.typing import ArrayLike
    
    RHS = Callable[[ArrayLike, ArrayLike], ArrayLike]
    T = TypeVar('T')
else:
    RHS = ''
    