from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import IsDone
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import LayoutDimension as D
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import BaseStyle, Style

T = TypeVar("T", bound=str | int)

lst = list


def style_from_dict(style_dict: dict[str, str]):
    # Deprecated function. Users should use Style.from_dict instead.
    # Keep this here for backwards-compatibility.
    return Style.from_dict(
        {key.lower(): value for key, value in style_dict.items()}
    )


def if_mousedown(handler: Callable[[MouseEvent], None]):
    def handle_if_mouse_down(mouse_event: MouseEvent):
        if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
            return handler(mouse_event)
        else:
            return NotImplemented

    return handle_if_mouse_down


class InquirerControl(FormattedTextControl, Generic[T]):
    def __init__(self, choices: Iterable[T], default: T | None, **kwargs):
        self.selected_option_index = 0
        self.answered = False
        self.choices: lst[tuple[T, T, None]] = []
        self._init_choices(choices, default)
        super().__init__(self._get_choice_tokens, **kwargs)

    def _init_choices(self, choices: Iterable[T], default: T | None):
        # helper to convert from question format to internal format
        searching_first_choice = True
        for i, c in enumerate(choices):
            self.choices.append((c, c, None))
            if searching_first_choice:
                self.selected_option_index = i  # found the first choice
                searching_first_choice = False
            if default is not None and default == c:
                self.selected_option_index = i  # default choice exists
                searching_first_choice = False

    @property
    def choice_count(self):
        return len(self.choices)

    def _get_choice_tokens(self):
        tokens = []

        def append(index: int, choice: tuple[T, T, None]):
            selected = index == self.selected_option_index

            @if_mousedown
            def select_item(mouse_event: MouseEvent):
                # bind option with this index to mouse event
                self.selected_option_index = index
                self.answered = True
                get_app().exit(result=self.get_selection()[0])

            tokens.append(
                (
                    "class:pointer" if selected else "",
                    " \u276f " if selected else "   ",
                )
            )
            if selected:
                tokens.append(("[SetCursorPosition]", ""))
            if choice[2]:  # disabled
                tokens.append(
                    (
                        "class:Selected" if selected else "",
                        "- %s (%s)" % (choice[0], choice[2]),
                    )
                )
            else:
                try:
                    tokens.append(
                        (
                            "class:Selected" if selected else "",
                            str(choice[0]),
                            select_item,
                        )
                    )
                except:
                    tokens.append(
                        (
                            "class:Selected" if selected else "",
                            choice[0],
                            select_item,
                        )
                    )
            tokens.append(("", "\n"))

        # prepare the select choices
        for i, choice in enumerate(self.choices):
            append(i, choice)
        tokens.pop()  # Remove last newline.
        return tokens

    def get_selection(self):
        return self.choices[self.selected_option_index]


def list(
    message: str,
    choices: Iterable[T],
    /,
    default: T | None = None,
    qmark: str = "[?]",
    style: BaseStyle | None = None,
    clear: bool = False,
) -> T:
    if clear:
        print("\033c", end="")

    ic = InquirerControl(choices, default=default)

    def get_prompt_tokens():
        tokens = []

        tokens.append(("class:questionmark", qmark))
        tokens.append(("class:question", " %s " % message))
        if ic.answered:
            tokens.append(("class:answer", f" {ic.get_selection()[0]}"))
        else:
            tokens.append(("class:instruction", " (Use arrow keys)"))
        return tokens

    # assemble layout
    layout = HSplit(
        [
            Window(
                height=D.exact(1),
                content=FormattedTextControl(get_prompt_tokens),
            ),
            ConditionalContainer(Window(ic), filter=~IsDone()),
        ]
    )

    # key bindings
    kb = KeyBindings()

    @kb.add("c-q", eager=True)
    @kb.add("c-c", eager=True)
    def _(event):
        raise KeyboardInterrupt()
        # event.app.exit(result=None)

    @kb.add("down", eager=True)
    def move_cursor_down(event):
        def _next():
            ic.selected_option_index = (
                ic.selected_option_index + 1
            ) % ic.choice_count

        _next()
        while ic.choices[ic.selected_option_index][2]:
            _next()

    @kb.add("up", eager=True)
    def move_cursor_up(event: KeyPressEvent):
        def _prev():
            ic.selected_option_index = (
                ic.selected_option_index - 1
            ) % ic.choice_count

        _prev()
        while ic.choices[ic.selected_option_index][2]:
            _prev()

    @kb.add("enter", eager=True)
    def set_answer(event: KeyPressEvent):
        ic.answered = True
        event.app.exit(result=ic.get_selection()[1])

    app = Application(
        layout=Layout(layout), key_bindings=kb, mouse_support=False, style=style
    )
    answer = app.run()
    return answer
