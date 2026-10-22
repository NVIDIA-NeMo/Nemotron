"""Regression tests for the helper scripts. Run: uv run pytest use-case-examples/french-public-service-assistant/tests"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_and_split import canonical_numbers, numbers_ok, route_ok  # noqa: E402
from build_seeds import extract_name, strip_name  # noqa: E402


def seed(professional: str, persona: str = "") -> dict:
    return {"professional_persona": professional, "persona": persona}


def test_name_prefix_with_extra_words_does_not_corrupt_text():
    # The text before the first comma runs past the name ("... qui gère les équipements, ...").
    text = "Marie Dupont qui gère les équipements et l'équipe municipale, aime la lecture."
    base = seed(text, "Marie incarne une retraitée.")
    assert extract_name(text) == "Marie Dupont"
    strip_name(base, extract_name(text))
    assert base["professional_persona"] == "l'usager qui gère les équipements et l'équipe municipale, aime la lecture."
    assert "équipements" in base["professional_persona"] and "équipe" in base["professional_persona"]
    assert base["persona"] == "l'usager incarne une retraitée."


def test_name_with_comma_and_name_parts_in_later_text():
    text = "Yvonne Hostin, ancienne secrétaire médicale. Hostin aime tricoter."
    base = seed(text)
    strip_name(base, extract_name(text))
    assert base["professional_persona"] == "l'usager, ancienne secrétaire médicale. l'usager aime tricoter."


def test_name_particles_and_short_words():
    assert extract_name("Jean de La Fontaine, ancien notaire") == "Jean de La Fontaine"
    base = seed("Jean de La Fontaine, ancien notaire de la ville.")
    strip_name(base, extract_name(base["professional_persona"]))
    assert base["professional_persona"] == "l'usager, ancien notaire de la ville."


def test_sentence_opening_is_not_a_name():
    assert extract_name("Anciennement agent, il vit à Lyon") is None
    assert extract_name("ancienne secrétaire, Yvonne") is None
    base = seed("Anciennement agent, il vit à Lyon")
    assert strip_name(base, None) is False
    assert base["professional_persona"] == "Anciennement agent, il vit à Lyon"


def test_name_after_job_description():
    text = "Ancien agent d'entretien voirie de la commune de Courlay, Rémi Perret allie rigueur et sens du service."
    assert extract_name(text) == "Rémi Perret"
    base = seed(text)
    strip_name(base, extract_name(text))
    assert base["professional_persona"] == (
        "Ancien agent d'entretien voirie de la commune de Courlay, l'usager allie rigueur et sens du service."
    )


def test_route_rejects_invented_route():
    assert not route_ok("Vous pouvez envoyer votre dossier par courrier à la mairie.")
    assert not route_ok("Utilisez la borne en mairie.")


def test_route_allows_negated_route():
    assert route_ok("Vous ne pouvez pas déposer votre demande par courrier : il faut aller au guichet.")
    assert route_ok("Le dépôt par courrier n'est pas possible. Prenez rendez-vous en mairie.")
    assert route_ok("Il est impossible de faire la demande sur papier.")


def test_negation_does_not_leak_across_sentences():
    assert not route_ok("Ne tardez pas. Vous pouvez déposer le dossier par courrier.")


def test_numbers_spelled_and_digit_forms_match():
    source = "un justificatif de domicile de un an ; photo de moins de 6 mois"
    assert numbers_ok("Apportez un justificatif de moins de 1 an et une photo de moins de 6 mois.", source)
    assert canonical_numbers("43,00 €") == {"43"}


def test_numbers_do_not_match_as_substrings():
    assert not numbers_ok("Comptez 2 semaines.", "Validité : 12 ans, retrait sous 3 mois")
    assert not numbers_ok("Le timbre coûte 86 €.", "Le timbre coûte 43 €.")
