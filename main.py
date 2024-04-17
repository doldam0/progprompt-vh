from typing import Literal


def f(a: int) -> tuple[Literal[True], int] | tuple[Literal[False], None]:
    if a > 0:
        return True, 1
    else:
        return False, None


success, num = f(1)
if success:
    print(num + 1)
else:
    print("No number")
