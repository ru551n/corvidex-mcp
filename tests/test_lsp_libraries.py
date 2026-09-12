"""Library inference for the generated vhdl_ls.toml.

vhdl_ls can only resolve ``<library>.<name>`` when ``<library>`` is
declared in the workspace configuration, so mapping files to the right
library is what makes library-qualified entity instantiation — the
standard tsfpga/hdl-modules style — navigable at all.
"""

from __future__ import annotations

import pytest

from corvidex_mcp.lsp.libraries import (
    DEFAULT_LIBRARY,
    group_by_library,
    infer_library,
    library_section,
    toml_string,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # The tsfpga layout: modules/<library>/<anything>.
        ("modules/cnn_accel/src/cnn_accel_pkg.vhd", "cnn_accel"),
        ("modules/cnn_accel/test/tb_cnn_accel_pool.vhd", "cnn_accel"),
        ("modules/cnn_accel/sim/foo.vhd", "cnn_accel"),
        ("modules/cnn_accel/rtl/foo.vhd", "cnn_accel"),
        ("modules/cnn_accel/regs_src/generated.vhd", "cnn_accel"),
        ("modules/cnn_accel/hdl/rtl/deep.vhd", "cnn_accel"),
        # A file directly in the module directory still belongs to it.
        ("modules/cnn_accel/top.vhd", "cnn_accel"),
        # A vendored checkout under a submodule prefix: its own
        # modules/ wins, so this is 'fifo', not 'hdl-modules'.
        ("hdl-modules/modules/fifo/src/asynchronous_fifo.vhd", "fifo"),
        ("vendor/hdl-modules/modules/common/rtl/attribute_pkg.vhd", "common"),
        # Windows separators resolve the same way.
        ("modules\\cnn_accel\\src\\cnn_accel_pkg.vhd", "cnn_accel"),
    ],
)
def test_infer_library_recognises_the_tsfpga_layout(path: str, expected: str) -> None:
    assert infer_library(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        # No modules/ segment at all.
        "rtl/top.vhd",
        "src/foo.vhd",
        "top.vhd",
        # modules/ but nothing below the candidate library directory.
        "modules/orphan.vhd",
        "modules",
        # Not usable as a library name.
        "modules/__pycache__/x.vhd",
        "modules/2020-archive/x.vhd",
        # Reserved: 'work' invalidates the entire vhdl_ls config, and
        # std/ieee are emitted separately for the bundled libraries.
        "modules/work/src/a.vhd",
        "modules/WORK/src/a.vhd",
        "modules/std/src/a.vhd",
        "modules/ieee/src/a.vhd",
    ],
)
def test_infer_library_returns_none_for_unrecognised_layouts(path: str) -> None:
    assert infer_library(path) is None


def test_group_by_library_splits_modules_and_keeps_the_rest_in_defaultlib() -> None:
    groups = group_by_library(
        [
            "rtl/top.vhd",
            "modules/cnn_accel/src/a.vhd",
            "hdl-modules/modules/fifo/src/b.vhd",
            "modules/cnn_accel/test/tb_a.vhd",
            "modules/work/src/c.vhd",
        ]
    )
    assert groups == {
        DEFAULT_LIBRARY: ["rtl/top.vhd", "modules/work/src/c.vhd"],
        "cnn_accel": ["modules/cnn_accel/src/a.vhd", "modules/cnn_accel/test/tb_a.vhd"],
        "fifo": ["hdl-modules/modules/fifo/src/b.vhd"],
    }
    # defaultlib first, then the inferred libraries alphabetically.
    assert list(groups) == [DEFAULT_LIBRARY, "cnn_accel", "fifo"]


def test_group_by_library_omits_defaultlib_when_everything_is_mapped() -> None:
    groups = group_by_library(["modules/cnn_accel/src/a.vhd"])
    assert list(groups) == ["cnn_accel"]


def test_group_by_library_of_nothing_is_empty() -> None:
    assert group_by_library([]) == {}


def test_toml_string_prefers_literal_strings() -> None:
    # Literal strings keep a Windows separator intact (no escaping).
    assert toml_string("modules\\a\\b.vhd") == "'modules\\a\\b.vhd'"
    assert toml_string("modules/a/b.vhd") == "'modules/a/b.vhd'"


def test_toml_string_escapes_what_a_literal_string_cannot_hold() -> None:
    assert toml_string("it's.vhd") == '"it\'s.vhd"'
    assert toml_string("a\nb.vhd") == '"a\\nb.vhd"'


def test_library_section_shape() -> None:
    assert library_section("fifo", ["a.vhd", "b.vhd"]) == [
        "[libraries.fifo]",
        "files = ['a.vhd', 'b.vhd']",
        "",
    ]
    assert library_section("ieee", ["x.vhdl"], third_party=True) == [
        "[libraries.ieee]",
        "files = ['x.vhdl']",
        "is_third_party = true",
        "",
    ]
