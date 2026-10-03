from bloodsmear.differential import summarize


def test_percentages_are_over_white_cells_only():
    labels = ["neutrophil"] * 6 + ["lymphocyte"] * 4 + ["erythroblast"] * 2
    d = summarize(labels, ["confident"] * len(labels))
    t = d.table.set_index("Cell type")
    assert d.counted_wbc == 10
    assert t.loc["Neutrophil", "Percent of white cells"] == 60.0
    assert d.nrbc_per_100_wbc == 20.0
    assert abs(t["Percent of white cells"].sum() - 100) < 1e-6


def test_uncertain_cells_are_held_back_not_counted():
    d = summarize(["neutrophil", "neutrophil", "basophil"], ["confident", "review", "unfamiliar"])
    assert d.counted_wbc == 1 and d.held_for_review == 2


def test_flags_left_shift_and_small_count():
    labels = ["neutrophil"] * 8 + ["immature_granulocyte"] * 2
    d = summarize(labels, ["confident"] * 10)
    text = " ".join(d.flags)
    assert "left shift" in text and "at least 100" in text


def test_a_single_immature_granulocyte_in_100_is_not_a_left_shift():
    labels = ["neutrophil"] * 60 + ["lymphocyte"] * 30 + ["monocyte"] * 5 + ["eosinophil"] * 4 + ["immature_granulocyte"]
    d = summarize(labels, ["confident"] * len(labels))
    assert not any("left shift" in f for f in d.flags)


def test_low_neutrophils_mention_local_ranges():
    labels = ["neutrophil"] * 30 + ["lymphocyte"] * 60 + ["monocyte"] * 7 + ["eosinophil"] * 3
    d = summarize(labels, ["confident"] * 100)
    assert any("Duffy-null" in f for f in d.flags)


def test_out_of_range_eosinophils_are_flagged():
    labels = ["neutrophil"] * 60 + ["lymphocyte"] * 25 + ["monocyte"] * 5 + ["eosinophil"] * 10
    d = summarize(labels, ["confident"] * 100)
    t = d.table.set_index("Cell type")
    assert t.loc["Eosinophil", "Flag"] == "Above range"
    assert t.loc["Neutrophil", "Flag"] == "Within range"
    assert not any("at least 100" in f for f in d.flags)


def test_empty_batch_does_not_crash():
    d = summarize(["erythroblast"], ["confident"])
    assert d.counted_wbc == 0 and d.nrbc_per_100_wbc is None
