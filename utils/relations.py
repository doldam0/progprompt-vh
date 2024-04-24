from enum import Enum
from typing import Literal, overload


class Relations(str, Enum):
    IS = "is"
    HOLD = "hold"
    INSIDE = "inside"
    ON = "on"
    CLOSE = "close"


RelationKeys = Literal["is", "hold", "inside", "on", "close"]


class States(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    ON = "on"
    OFF = "off"
    PLUGGED_IN = "plugged_in"
    PLUGGED_OUT = "plugged_out"


StateKeys = Literal["open", "closed", "on", "off", "plugged_in", "plugged_out"]


class Relation:
    left: str
    relation: Relations
    right: str

    def __init__(self, left: str, relation: Relations, right: str) -> None:
        self.left = left
        self.relation = relation
        self.right = right

    def __repr__(self) -> str:
        return f"{self.left} {self.relation} {self.right}"


class StateRelation(Relation):
    right: States

    def __init__(self, object: str, state: States) -> None:
        super().__init__(object, Relations.IS, state)

    @property
    def object(self) -> str:
        return self.left

    @property
    def state(self) -> States:
        return self.right


class HoldsRelation(Relation):
    def __init__(self, object: str) -> None:
        super().__init__("agent", Relations.HOLD, object)

    @property
    def object(self) -> str:
        return self.right


class InsideRelation(Relation):
    def __init__(self, object: str, container: str) -> None:
        super().__init__(object, Relations.INSIDE, container)

    @property
    def container(self) -> str:
        return self.right

    @property
    def object(self) -> str:
        return self.left


@overload
def relate(
    left: str, relation: Literal["is"], right: StateKeys
) -> StateRelation: ...


@overload
def relate(
    left: Literal["agent"], relation: Literal["hold"], right: str
) -> HoldsRelation: ...


@overload
def relate(
    left: str, relation: Literal["inside"], right: str
) -> InsideRelation: ...


@overload
def relate(
    left: str, relation: Literal["on", "close"], right: str
) -> Relation: ...


def relate(left: str, relation: RelationKeys, right: str) -> Relation:
    if relation == Relations.IS:
        return StateRelation(left, States[right])
    if relation == Relations.HOLD:
        return HoldsRelation(right)
    if relation == Relations.INSIDE:
        return InsideRelation(left, right)
    return Relation(left, Relations[relation], right)
