"""Regra de decisão da busca por coordenadas (Etapa 2)."""
from nslkdd.search import AXES, FAMILIES, pick


def _r(mean, std=0.002):
    return {"f1_macro_mean": mean, "f1_macro_std": std}


def test_pick_keeps_reference_within_one_std():
    options = [{"x": "simples"}, {"x": "complexa"}]
    chosen, _ = pick([_r(0.970), _r(0.9715)], options)
    assert chosen == {"x": "simples"}


def test_pick_switches_when_gain_exceeds_one_std():
    options = [{"x": "a"}, {"x": "b"}, {"x": "c"}]
    chosen, best = pick([_r(0.970), _r(0.975), _r(0.973)], options)
    assert chosen == {"x": "b"} and best["f1_macro_mean"] == 0.975


def test_first_option_of_each_axis_is_the_reference():
    for family, axes in AXES.items():
        assert family in FAMILIES
        for _, options in axes:
            assert options[0] and len(options) >= 2
