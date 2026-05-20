from inspect_ai import Task
from tasks.consultancy import (
    animal_cascade_scenarios,
)
from animal_cascades.graders import (
    UTILITY_TEMPLATE,
    ETHICS_TEMPLATE,
)


def test_task_returns_task_instance():
    t = animal_cascade_scenarios()
    assert isinstance(t, Task)

def test_utility_template_has_required_placeholders():
    assert "{question}" in UTILITY_TEMPLATE
    assert "{answer}" in UTILITY_TEMPLATE
    assert "{instructions}" in UTILITY_TEMPLATE


def test_ethics_template_has_required_placeholders():
    assert "{question}" in ETHICS_TEMPLATE
    assert "{answer}" in ETHICS_TEMPLATE
    assert "{instructions}" in ETHICS_TEMPLATE


def test_utility_template_excludes_animal_welfare_framing():
    assert "animal welfare" not in UTILITY_TEMPLATE.lower()


def test_utility_template_covers_six_components():
    for term in ["problem focus", "innovation", "actionability", "cost", "timeline", "value"]:
        assert term in UTILITY_TEMPLATE.lower(), f"utility template missing component: {term}"


def test_ethics_template_covers_five_components():
    for term in ["sentience", "cruelty", "intrinsic", "rights", "ecological"]:
        assert term in ETHICS_TEMPLATE.lower(), f"ethics template missing component: {term}"
