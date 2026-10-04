"""Explicit alignment to the independently reconstructed source table.

Only names, age bins and answer representations are aligned here; the second
table is not generated from the primary rule cards. Unknowns are enumerated by
the primary domain and every complete world is checked independently.
"""
try:
    from .independent_oracle import complete_outcome, load_rules
except ImportError:
    from independent_oracle import complete_outcome, load_rules

SECOND = load_rules()


def independent_value(key, f):
    if key == 'tk:92':
        name='tk92_age_weekly'; x=dict(age_band='under16' if f['age']<16 else '16to17', general_or_secondary_vocational_education=True, combines_during_academic_year=f['study'])
    elif key == 'tk:93':
        name='tk93_mandatory_part_time'; x=dict(request=f['request'],pregnant=False,qualifying_one_parent_guardian_trustee=True,child_under14=f['child_age']<14,disabled_child_under18=f['disabled'] and f['child_age']<18,cares_for_sick_family_member=False,compliant_medical_opinion=False)
    elif key == 'tk:94':
        name='tk94_age_daily'; x=dict(age_band=str(f['age']) if f['age']<16 else '16to17',general_or_secondary_vocational_education=True,combines_during_academic_year=f['study'])
    elif key == 'tk:128':
        name='tk128_named_ground_cap'; x=dict(written_application=f['written'],ground=dict(war='ww2_participant',pension='old_age_pensioner_worker',disabled='disabled_worker',birth='child_birth',marriage='marriage_registration',death='close_relative_death')[f['ground']])
    elif key == 'tk:173':
        name='tk173_intermediate_attestation'; x=dict(accredited_bachelor_specialist_master=True,study_form=dict(part_time='correspondence',evening='part_time',full_time='full_time')[f['form']],successful_study=True,first_same_level=True,repeat_employer_written_contract=False,selected_organization_if_two=True,course={1:'first',2:'second',3:'later'}[f['course']],shortened_program=f['accelerated'])
    elif key == 'sk:19':
        name='sk19_registry_grounds'; x=dict(mutual_consent=f['consent'],common_minor_children=f['children'],one_spouse_application=True,other_spouse_status=dict(none='ordinary',missing='court_missing',incapable='court_incapable',prison_2='imprisoned_up_to3',prison_3='imprisoned_up_to3',prison_4='imprisoned_over3')[f['status']])
    elif key == 'sk:57':
        name='sk57_mandatory_opinion'; x=dict(age='under10' if f['age']<10 else '10to17',contrary_to_child_interests=f['against'])
    elif key == 'sk:81':
        name='sk81_standard_fraction'; x=dict(no_alimony_agreement=not f['agreement'],minor_children='one' if f['children']==1 else 'two' if f['children']==2 else 'three_plus')
    elif key == 'gk:26':
        decision=f['decision']
        if decision=='restricted': decision='restricted_transaction_within_restriction' if f['within_restriction'] else 'restricted_transaction_outside_restriction'
        name='gk26_own_income'; x=dict(age='16to17',own_income=True,full_capacity_basis='none',court_income_decision=decision)
    elif key == 'gk:28':
        name='gk28_gratuitous_benefit'; x=dict(age='under6' if f['age']<6 else '6to13',gratuitous_receipt_of_benefit=True,requires_notarization=f['notary'],requires_state_registration=f['registration'])
    elif key == 'gk:37':
        name='gk37_3_exception'; x=dict(counterparty='guardian',transfer=(f['kind']+('_to_ward' if f['to_ward'] else '_from_ward')) if f['kind']!='sale' else 'other_transaction')
    elif key == 'gk:186':
        name='gk186_default_term'; x=dict(date_present=f['date'],term_present=False,notarized=f['notary'],for_actions_abroad=f['abroad'])
    elif key == '59:8':
        name='fz59_art8_registration_redirect'; x=dict(operation='redirection',readable_text=f['readable'],essence_identifiable=True,migration_facts=f['migration'],outside_recipient_competence=not f['competent'],complaint_route_forbidden=False)
    elif key == '59:11':
        name='fz59_art11_5_correspondence'; x=dict(repeated_written_substantive_answers=f['repeated'],new_arguments_or_circumstances=f['new'],same_body_or_official_for_all=f['same'],authorized_decision_maker=True)
    elif key == '59:12':
        name='fz59_art12_base_term'; x=dict(within_competence=True,highest_regional_official_recipient=f['head'],migration_facts=f['migration'])
    else:
        raise ValueError(key)
    result = complete_outcome(SECOND[name], x)
    if isinstance(result, dict):
        for field in ['max_hours_per_week','max_hours_per_day','up_to_calendar_days','calendar_days','fraction','within_days']:
            if field in result: return str(result[field])
        if 'duration' in result:
            return {'until_revoked_by_issuer':'до отмены','one_year_from_execution':'1 год'}[result['duration']]
        if result.get('may_decide_groundless_and_end_correspondence'): return 'да'
    yes={'employer_obliged_under_part2','registry_ground_p1','registry_ground_p2','mandatory_consideration','independent_under26_2_1','independent_under28_2_2','within_transaction_exception'}
    no={'obligation_not_established_under_part2','registry_ground_not_established_by_p1_p2','mandatory_consideration_not_established_by_sentence2','not_independent_for_this_disposition','outside_28_2_age','independence_not_established_by28_2_2','prohibited_by37_3','permission_not_established_under11_5'}
    outside={'mandatory_grant_not_triggered','part1_guarantee_not_established','outside_p1','no_redirection_under11_4','no_redirection_duty_established_by8_3_3_1'}
    if result in yes: return 'да'
    if result in no: return 'нет'
    if result in outside: return 'не следует из указанной нормы'
    if result=='void_no_execution_date': return 'ничтожна'
    raise ValueError(f'Unaligned independent answer {key}: {result}')
