from enum import Enum
from typing import Literal, overload


class Relations(str, Enum):
    IS = "IS"
    HOLD = "HOLD"
    INSIDE = "INSIDE"
    ON = "ON"
    CLOSE = "CLOSE"


RelationKeys = Literal["IS", "HOLD", "INSIDE", "ON", "CLOSE"]


class States(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ON = "ON"
    OFF = "OFF"
    PLUGGED_IN = "PLUGGED_IN"
    PLUGGED_OUT = "PLUGGED_OUT"


StateKeys = Literal["OPEN", "CLOSED", "ON", "OFF", "PLUGGED_IN", "PLUGGED_OUT"]


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
    left: str, relation: Literal["IS"], right: StateKeys
) -> StateRelation: ...


@overload
def relate(
    left: Literal["agent"], relation: Literal["HOLD"], right: str
) -> HoldsRelation: ...


@overload
def relate(
    left: str, relation: Literal["INSIDE"], right: str
) -> InsideRelation: ...


@overload
def relate(
    left: str, relation: Literal["ON", "CLOSE"], right: str
) -> Relation: ...


def relate(left: str, relation: RelationKeys, right: str) -> Relation:
    if relation == Relations.IS:
        return StateRelation(left, States[right])
    if relation == Relations.HOLD:
        return HoldsRelation(right)
    if relation == Relations.INSIDE:
        return InsideRelation(left, right)
    return Relation(left, Relations[relation], right)
