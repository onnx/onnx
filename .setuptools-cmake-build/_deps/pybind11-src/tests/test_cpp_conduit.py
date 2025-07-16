# Copyright (c) 2024 The pybind Community.

from __future__ import annotations

import exo_planet_c_api
import exo_planet_pybind11
import home_planet_very_lonely_traveler
import pytest

from pybind11_tests import cpp_conduit as home_planet


def test_traveler_getattr_actually_exists():
    t_h = home_planet.Traveler("home")
    assert t_h.any_name == "Traveler GetAttr: any_name luggage: home"


def test_premium_traveler_getattr_actually_exists():
    t_h = home_planet.PremiumTraveler("home", 7)
    assert t_h.secret_name == "PremiumTraveler GetAttr: secret_name points: 7"


def test_call_cpp_conduit_success():
    t_h = home_planet.Traveler("home")
    cap = t_h._pybind11_conduit_v1_(
        home_planet.PYBIND11_PLATFORM_ABI_ID,
        home_planet.cpp_type_info_capsule_Traveler,
        b"raw_pointer_ephemeral",
    )
    assert cap.__class__.__name__ == "PyCapsule"


def test_call_cpp_conduit_platform_abi_id_mismatch():
    t_h = home_planet.Traveler("home")
    cap = t_h._pybind11_conduit_v1_(
        home_planet.PYBIND11_PLATFORM_ABI_ID + b"MISMATCH",
        home_planet.cpp_type_info_capsule_Traveler,
        b"raw_pointer_ephemeral",
    )
    assert cap is None


def test_call_cpp_conduit_cpp_type_info_capsule_mismatch():
    t_h = home_planet.Traveler("home")
    cap = t_h._pybind11_conduit_v1_(
        home_planet.PYBIND11_PLATFORM_ABI_ID,
        home_planet.cpp_type_info_capsule_int,
        b"raw_pointer_ephemeral",
    )
    assert cap is None


def test_call_cpp_conduit_pointer_kind_invalid():
    t_h = home_planet.Traveler("home")
    with pytest.raises(
        RuntimeError, match='^Invalid pointer_kind: "raw_pointer_ephemreal"$'
    ):
        t_h._pybind11_conduit_v1_(
            home_planet.PYBIND11_PLATFORM_ABI_ID,
            home_planet.cpp_type_info_capsule_Traveler,
            b"raw_pointer_ephemreal",
        )


def test_home_only_basic():
    t_h = home_planet.Traveler("home")
    assert t_h.luggage == "home"
    assert home_planet.get_luggage(t_h) == "home"


def test_home_only_premium():
    p_h = home_planet.PremiumTraveler("home", 2)
    assert p_h.luggage == "home"
    assert home_planet.get_luggage(p_h) == "home"
    assert home_planet.get_points(p_h) == 2


def test_exo_only_basic():
    t_e = exo_planet_pybind11.Traveler("exo")
    assert t_e.luggage == "exo"
    assert exo_planet_pybind11.get_luggage(t_e) == "exo"


def test_exo_only_premium():
    p_e = exo_planet_pybind11.PremiumTraveler("exo", 3)
    assert p_e.luggage == "exo"
    assert exo_planet_pybind11.get_luggage(p_e) == "exo"
    assert exo_planet_pybind11.get_points(p_e) == 3


def test_home_passed_to_exo_basic():
    t_h = home_planet.Traveler("home")
    assert exo_planet_pybind11.get_luggage(t_h) == "home"


def test_exo_passed_to_home_basic():
    t_e = exo_planet_pybind11.Traveler("exo")
    assert home_planet.get_luggage(t_e) == "exo"


def test_home_passed_to_exo_premium():
    p_h = home_planet.PremiumTraveler("home", 2)
    assert exo_planet_pybind11.get_luggage(p_h) == "home"
    assert exo_planet_pybind11.get_points(p_h) == 2


def test_exo_passed_to_home_premium():
    p_e = exo_planet_pybind11.PremiumTraveler("exo", 3)
    assert home_planet.get_luggage(p_e) == "exo"
    assert home_planet.get_points(p_e) == 3


@pytest.mark.parametrize(
    "traveler_type", [home_planet.Traveler, exo_planet_pybind11.Traveler]
)
def test_exo_planet_c_api_traveler(traveler_type):
    t = traveler_type("socks")
    assert exo_planet_c_api.GetLuggage(t) == "socks"


@pytest.mark.parametrize(
    "premium_traveler_type",
    [home_planet.PremiumTraveler, exo_planet_pybind11.PremiumTraveler],
)
def test_exo_planet_c_api_premium_traveler(premium_traveler_type):
    pt = premium_traveler_type("gucci", 5)
    assert exo_planet_c_api.GetLuggage(pt) == "gucci"
    assert exo_planet_c_api.GetPoints(pt) == 5


def test_home_planet_wrap_very_lonely_traveler():
    # This does not exercise the cpp_conduit feature, but is here to
    # demonstrate that the cpp_conduit feature does not solve all
    # cross-extension interoperability issues.
    # Here is the proof that the following works for extensions with
    # matching `PYBIND11_INTERNALS_ID`s:
    #     test_cpp_conduit.cpp:
    #         py::class_<LonelyTraveler>
    #     home_planet_very_lonely_traveler.cpp:
    #         py::class_<VeryLonelyTraveler, LonelyTraveler>
    # See test_exo_planet_pybind11_wrap_very_lonely_traveler() for the negative
    # test.
    assert home_planet.LonelyTraveler is not None  # Verify that the base class exists.
    home_planet_very_lonely_traveler.wrap_very_lonely_traveler()
    # Ensure that the derived class exists.
    assert home_planet_very_lonely_traveler.VeryLonelyTraveler is not None


def test_exo_planet_pybind11_wrap_very_lonely_traveler():
    # See comment under test_home_planet_wrap_very_lonely_traveler() first.
    # Here the `PYBIND11_INTERNALS_ID`s don't match between:
    #     test_cpp_conduit.cpp:
    #         py::class_<LonelyTraveler>
    #     exo_planet_pybind11.cpp:
    #         py::class_<VeryLonelyTraveler, LonelyTraveler>
    assert home_planet.LonelyTraveler is not None  # Verify that the base class exists.
    with pytest.raises(
        RuntimeError,
        match='^generic_type: type "VeryLonelyTraveler" referenced unknown base type '
        '"pybind11_tests::test_cpp_conduit::LonelyTraveler"$',
    ):
        exo_planet_pybind11.wrap_very_lonely_traveler()
