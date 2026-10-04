"""Independent reconstruction of 15 new rules and five demonstration rules.

The functions were reconstructed from archived official source text without
reading expansion cards, their generator, or their gold answers. They express
only the narrow consequences documented in expansion_source_review.json.
They do not decide whole cases or certify the law in force on a particular date.

Functions consume complete, legally qualified facts. ``evaluate_partial`` can
lift them to missing facts, but its finite domains must be supplied explicitly;
an unreported fact is never silently treated as false.
"""

from inspect import signature
from itertools import product
from typing import Callable, Mapping, Sequence


OUTSIDE_SCOPE = "outside_declared_scope"
NO_REQUIREMENT = "requirement_not_established_by_selected_provision"
NO_REWARD_RIGHT = "no_reward_right_under_article_229"
AGREEMENT_REQUIRED = "reward_amount_requires_agreement"
DIFFERENT_RISK_RULE = "other_risk_rule_must_be_examined"
INSUFFICIENT_FACTS = "insufficient_facts"
INCOMPATIBLE_FACTS = "incompatible_facts"


class InvalidFacts(ValueError):
    """The supplied facts are outside their types or mutually incompatible."""


def _booleans(**facts: bool) -> None:
    for name, value in facts.items():
        if type(value) is not bool:
            raise InvalidFacts(f"{name} must be a boolean, not a missing fact")


def _nonnegative_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise InvalidFacts(f"{name} must be a nonnegative integer")


def tk67_contract_deemed_concluded(
    *,
    actually_started_work: bool,
    employer_knew_or_directed: bool,
    authorized_representative_knew_or_directed: bool,
) -> bool:
    """Article67(2), first sentence; no written employment contract exists."""
    _booleans(
        actually_started_work=actually_started_work,
        employer_knew_or_directed=employer_knew_or_directed,
        authorized_representative_knew_or_directed=authorized_representative_knew_or_directed,
    )
    return actually_started_work and (
        employer_knew_or_directed or authorized_representative_knew_or_directed
    )


def tk79_minimum_written_notice_days(
    *, fixed_term_replaces_absent_employee: bool
) -> int | str:
    """Expiry termination; return minimum calendar days, not working days."""
    _booleans(fixed_term_replaces_absent_employee=fixed_term_replaces_absent_employee)
    return NO_REQUIREMENT if fixed_term_replaces_absent_employee else 3


def tk122_mandatory_early_leave(
    *,
    employee_applied: bool,
    employee_age_years: int,
    woman_before_maternity_leave: bool,
    woman_immediately_after_maternity_leave: bool,
    qualifying_adoption_of_child_under_three_months: bool,
) -> bool:
    """First employment year before six continuous months; other grounds excluded.

    The adoption fact is a supplied qualification under the under-three-month
    clause. It does not mean an adoption years ago when the child was an infant.
    """
    _nonnegative_integer("employee_age_years", employee_age_years)
    _booleans(
        employee_applied=employee_applied,
        woman_before_maternity_leave=woman_before_maternity_leave,
        woman_immediately_after_maternity_leave=woman_immediately_after_maternity_leave,
        qualifying_adoption_of_child_under_three_months=qualifying_adoption_of_child_under_three_months,
    )
    if employee_age_years < 18 and qualifying_adoption_of_child_under_three_months:
        raise InvalidFacts("Family Code127(1) requires an adult adopter")
    if woman_before_maternity_leave and woman_immediately_after_maternity_leave:
        raise InvalidFacts("The proposed leave cannot precede and follow the same maternity leave")
    qualifies = (
        employee_age_years < 18
        or woman_before_maternity_leave
        or woman_immediately_after_maternity_leave
        or qualifying_adoption_of_child_under_three_months
    )
    return employee_applied and qualifies


def tk124_nonprovision_expressly_prohibited(
    *,
    consecutive_years_without_annual_leave: int,
    employee_age_years: int,
    works_in_harmful_or_dangerous_conditions: bool,
) -> bool:
    """Test the last paragraph, assuming actual/proposed nonprovision this year."""
    _nonnegative_integer("consecutive_years_without_annual_leave", consecutive_years_without_annual_leave)
    _nonnegative_integer("employee_age_years", employee_age_years)
    _booleans(works_in_harmful_or_dangerous_conditions=works_in_harmful_or_dangerous_conditions)
    if consecutive_years_without_annual_leave == 0:
        raise InvalidFacts("A nonprovision question requires at least the current year")
    return (
        consecutive_years_without_annual_leave >= 2
        or employee_age_years < 18
        or works_in_harmful_or_dangerous_conditions
    )


def tk126_basic_leave_cash_substitution_permitted(
    *,
    basic_leave_days_for_one_working_year: int,
    days_proposed_for_cash_substitution: int,
    employee_written_request: bool,
    pregnant: bool,
    employee_age_years: int,
) -> bool:
    """General permission, not an employer duty; no dismissal or special exception.

    Only annual BASIC leave for one working year is modeled. Harmful-work
    additional leave, accumulated years, and special Code exceptions are out.
    """
    for name, value in (
        ("basic_leave_days_for_one_working_year", basic_leave_days_for_one_working_year),
        ("days_proposed_for_cash_substitution", days_proposed_for_cash_substitution),
        ("employee_age_years", employee_age_years),
    ):
        _nonnegative_integer(name, value)
    _booleans(employee_written_request=employee_written_request, pregnant=pregnant)
    if basic_leave_days_for_one_working_year < 28:
        raise InvalidFacts("The declared ordinary annual basic-leave scope starts at 28 days")
    if days_proposed_for_cash_substitution == 0:
        raise InvalidFacts("The question concerns substitution of a positive number of days")
    if employee_age_years < 18 and basic_leave_days_for_one_working_year < 31:
        raise InvalidFacts("Article 267 establishes 31 basic days for a minor")
    return (
        employee_written_request
        and not pregnant
        and employee_age_years >= 18
        and days_proposed_for_cash_substitution <= basic_leave_days_for_one_working_year - 28
    )


def sk35_notarized_spousal_consent_required(
    *,
    disposed_property_rights_require_state_registration: bool,
    transaction_has_mandatory_notarial_form: bool,
    transaction_requires_state_registration: bool,
) -> bool:
    """Article35(3), first sentence; transaction disposes of spouses' common property."""
    _booleans(
        disposed_property_rights_require_state_registration=disposed_property_rights_require_state_registration,
        transaction_has_mandatory_notarial_form=transaction_has_mandatory_notarial_form,
        transaction_requires_state_registration=transaction_requires_state_registration,
    )
    return (
        disposed_property_rights_require_state_registration
        or transaction_has_mandatory_notarial_form
        or transaction_requires_state_registration
    )


def sk62_independent_parental_rights(
    *,
    parent_age_years: int,
    parent_is_unmarried: bool,
    child_born: bool,
    this_parents_maternity_or_paternity_established: bool,
) -> bool | str:
    """Article62(2), first sentence only; not all rights of minor parents."""
    _nonnegative_integer("parent_age_years", parent_age_years)
    _booleans(
        parent_is_unmarried=parent_is_unmarried,
        child_born=child_born,
        this_parents_maternity_or_paternity_established=this_parents_maternity_or_paternity_established,
    )
    if parent_age_years >= 18 or not parent_is_unmarried:
        return OUTSIDE_SCOPE
    if not child_born and this_parents_maternity_or_paternity_established:
        raise InvalidFacts("Established parentage here concerns a born child")
    return (
        parent_age_years >= 16
        and child_born
        and this_parents_maternity_or_paternity_established
    )


def gk211_accidental_loss_risk_bearer(
    *, different_statutory_rule_applies: bool, different_contractual_rule_applies: bool
) -> str:
    """Accidental loss or damage, not liability for culpable harm."""
    _booleans(
        different_statutory_rule_applies=different_statutory_rule_applies,
        different_contractual_rule_applies=different_contractual_rule_applies,
    )
    if different_statutory_rule_applies or different_contractual_rule_applies:
        return DIFFERENT_RISK_RULE
    return "owner"


def gk214_state_ownership_from_paragraph2(
    *,
    land_or_other_natural_resource: bool,
    owned_by_citizens: bool,
    owned_by_legal_entities: bool,
    owned_by_municipality: bool,
) -> bool | str:
    """Category of ownership only; no allocation between federation and region."""
    _booleans(
        land_or_other_natural_resource=land_or_other_natural_resource,
        owned_by_citizens=owned_by_citizens,
        owned_by_legal_entities=owned_by_legal_entities,
        owned_by_municipality=owned_by_municipality,
    )
    if not land_or_other_natural_resource:
        return OUTSIDE_SCOPE
    return not (owned_by_citizens or owned_by_legal_entities or owned_by_municipality)


def gk227_finder_sale_meets_selected_conditions(
    *,
    perishable: bool,
    storage_cost_disproportionate_to_value: bool,
    sale_includes_obtaining_written_proceeds_evidence: bool,
) -> bool:
    """Finding kept lawfully; compare permission with the required evidence step.

    The evidence is obtained with realization. It need not pre-exist the sale.
    Other duties of notice and return are not displaced by a True result.
    """
    _booleans(
        perishable=perishable,
        storage_cost_disproportionate_to_value=storage_cost_disproportionate_to_value,
        sale_includes_obtaining_written_proceeds_evidence=sale_includes_obtaining_written_proceeds_evidence,
    )
    return (
        perishable or storage_cost_disproportionate_to_value
    ) and sale_includes_obtaining_written_proceeds_evidence


def gk229_reward_upper_percentage(
    *, finding_reported: bool, attempted_concealment: bool, valuable_only_to_entitled_recipient: bool
) -> int | str:
    """Return statutory20% ceiling, no reward right, or agreement-only branch."""
    _booleans(
        finding_reported=finding_reported,
        attempted_concealment=attempted_concealment,
        valuable_only_to_entitled_recipient=valuable_only_to_entitled_recipient,
    )
    if not finding_reported or attempted_concealment:
        return NO_REWARD_RIGHT
    if valuable_only_to_entitled_recipient:
        return AGREEMENT_REQUIRED
    return 20


def gk234_minimum_prescription_period_reached(
    *, immovable_property: bool, qualifying_completed_possession_years: int
) -> bool:
    """General5/15-year period only; all other conditions and start date stipulated."""
    _booleans(immovable_property=immovable_property)
    _nonnegative_integer("qualifying_completed_possession_years", qualifying_completed_possession_years)
    minimum = 15 if immovable_property else 5
    return qualifying_completed_possession_years >= minimum


def fz5_material_inspection_right(
    *, related_to_consideration_of_this_appeal: bool, affects_others_rights: bool, contains_protected_secret: bool
) -> bool:
    """Article5(2); no decision on partial/redacted access under other rules."""
    _booleans(
        related_to_consideration_of_this_appeal=related_to_consideration_of_this_appeal,
        affects_others_rights=affects_others_rights,
        contains_protected_secret=contains_protected_secret,
    )
    return related_to_consideration_of_this_appeal and not affects_others_rights and not contains_protected_secret


def fz6_disclosure_barred_by_part2(
    *, citizen_consented: bool, written_appeal_forwarded_to_competent_recipient: bool
) -> bool:
    """Information in the appeal; competent forwarding is not disclosure here.

    A False answer does not authorize disclosure under all other applicable law.
    It does not cover unrelated private information added to a forwarded appeal.
    """
    _booleans(
        citizen_consented=citizen_consented,
        written_appeal_forwarded_to_competent_recipient=written_appeal_forwarded_to_competent_recipient,
    )
    return not citizen_consented and not written_appeal_forwarded_to_competent_recipient


def fz7_paper_appeal_required_details_complete(
    *,
    addressee_body_named: bool,
    addressee_official_full_name: bool,
    addressee_official_position: bool,
    sender_surname: bool,
    sender_given_name: bool,
    sender_has_patronymic: bool,
    sender_patronymic_given: bool,
    postal_reply_address: bool,
    substance_stated: bool,
    personal_signature: bool,
    date_given: bool,
) -> bool:
    """Part1 paper appeal by a citizen; not the legal consequence of an omission."""
    facts = locals()
    _booleans(**facts)
    if not sender_has_patronymic and sender_patronymic_given:
        raise InvalidFacts("A sender cannot supply their own nonexistent patronymic")
    recipient = addressee_body_named or addressee_official_full_name or addressee_official_position
    own_name = sender_surname and sender_given_name and (not sender_has_patronymic or sender_patronymic_given)
    return (
        recipient
        and own_name
        and postal_reply_address
        and substance_stated
        and personal_signature
        and date_given
    )


def demo_tk115_ordinary_basic_leave_days(*, ordinary_unextended_leave: bool) -> int | str:
    _booleans(ordinary_unextended_leave=ordinary_unextended_leave)
    return 28 if ordinary_unextended_leave else OUTSIDE_SCOPE


def demo_tk267_minor_basic_leave_days(*, employee_age_years: int) -> int | str:
    _nonnegative_integer("employee_age_years", employee_age_years)
    return 31 if employee_age_years < 18 else OUTSIDE_SCOPE


def demo_sk56_child_may_apply_to_court(
    *, child_age_years: int, child_rights_or_legal_interests_violated: bool
) -> bool | str:
    """Part2; child without separately acquired full capacity; no whole-case ruling."""
    _nonnegative_integer("child_age_years", child_age_years)
    _booleans(child_rights_or_legal_interests_violated=child_rights_or_legal_interests_violated)
    if child_age_years >= 18:
        return OUTSIDE_SCOPE
    return child_rights_or_legal_interests_violated and child_age_years >= 14


def demo_gk32_adult_guardianship(
    *, adult: bool, court_declared_incapable_due_to_mental_disorder: bool
) -> str:
    _booleans(
        adult=adult,
        court_declared_incapable_due_to_mental_disorder=court_declared_incapable_due_to_mental_disorder,
    )
    if not adult:
        return OUTSIDE_SCOPE
    return "guardianship" if court_declared_incapable_due_to_mental_disorder else NO_REQUIREMENT


def demo_gk33_adult_trusteeship(*, adult: bool, court_limited_legal_capacity: bool) -> str:
    _booleans(adult=adult, court_limited_legal_capacity=court_limited_legal_capacity)
    if not adult:
        return OUTSIDE_SCOPE
    return "trusteeship" if court_limited_legal_capacity else NO_REQUIREMENT


def evaluate_partial(
    function: Callable,
    facts: Mapping[str, object],
    unknown_domains: Mapping[str, Sequence[object]],
) -> object:
    """Unanimity over explicitly supplied finite completions of absent/None facts.

    The caller must establish that each finite domain represents all legally
    relevant possibilities under its stated scope. InvalidFacts completions are
    excluded. Empty domains or contradictory complete facts yield incompatibility.
    No domains are inferred from a dataset, generator, or another oracle.
    """
    keys = list(signature(function).parameters)
    if set(facts) - set(keys) or set(unknown_domains) - set(keys):
        raise ValueError("Unknown fact or domain key")
    choices = []
    for key in keys:
        value = facts.get(key)
        if value is None:
            if key not in unknown_domains:
                raise ValueError(f"An explicit completion domain is required for {key}")
            choices.append(unknown_domains[key])
        else:
            choices.append([value])
    results = {}
    for values in product(*choices):
        try:
            answer = function(**dict(zip(keys, values)))
        except InvalidFacts:
            continue
        results[(type(answer).__name__, answer)] = answer
    if not results:
        return INCOMPATIBLE_FACTS
    if len(results) != 1:
        return INSUFFICIENT_FACTS
    return next(iter(results.values()))
