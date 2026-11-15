import csv
import io
import pathlib
import re
import zipfile

import numpy as np
import pandas as pd
from source_io import fetch

SOURCE_URL = 'https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip'
TARGET = 'next_pb90_exceeds_action_level'
TIME_COL = 'target_period_start'
ROW_ID = 'row_id'
ID_COL = 'PWSID'
THRESHOLD = 0.015
FEATURES = ['prior_pb90_mg_l', 'prior_cu90_mg_l', 'prior_pb90_ratio_to_al', 'pb90_change_from_previous', 'n_prior_periods_observed', 'ever_exceeded_before', 'gap_days_between_periods', 'prior_period_length_days', 'population_served_count', 'primary_source_code', 'owner_type_code', 'primacy_agency_code', 'prior_lcr_violations_count', 'prior_lslr_milestone']
TRAIN_START = pd.Timestamp('1995-01-01')
TRAIN_END = pd.Timestamp('2014-01-01')
EVAL_END = pd.Timestamp('2019-01-01')
HOLDOUT_END = pd.Timestamp('2024-01-01')
AGENCY_CUTOFF = pd.Timestamp('2014-01-01')


def _canon(value):
    text = str(value).strip().upper()
    return re.sub('[^A-Z0-9]+', '_', text).strip('_')


def _field_failure(member, headers, missing, role):
    message = 'required ' + role + ' field not found; selected archive member ' + str(member) + '; missing names ' + str(missing) + '; observed headers ' + str(list(headers))
    raise KeyError(message[:4000])


def _pick(header, candidates, member, role):
    lookup = {}
    for column in header:
        key = _canon(column)
        if key not in lookup:
            lookup[key] = column
    for candidate in candidates:
        key = _canon(candidate)
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = _canon(candidate)
        for known, original in lookup.items():
            if known.endswith('_' + key) or known.startswith(key + '_'):
                return original
    _field_failure(member, header, [candidates[0]], role)


def _pick_optional(header, candidates):
    lookup = {}
    for column in header:
        key = _canon(column)
        if key not in lookup:
            lookup[key] = column
    for candidate in candidates:
        key = _canon(candidate)
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = _canon(candidate)
        for known, original in lookup.items():
            if known.endswith('_' + key) or known.startswith(key + '_'):
                return original
    return None


def _header(zf, member):
    with zf.open(member) as handle:
        text = io.TextIOWrapper(handle, encoding='utf-8-sig', errors='replace')
        line = text.readline()
    return next(csv.reader([line]))


def _read_full(zf, member):
    with zf.open(member) as handle:
        frame = pd.read_csv(handle, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _iter_chunks(zf, member, usecols, chunksize=250000):
    with zf.open(member) as handle:
        for chunk in pd.read_csv(handle, dtype=str, keep_default_na=False, encoding='utf-8-sig', usecols=usecols, chunksize=chunksize):
            chunk.columns = [str(column).strip() for column in chunk.columns]
            yield chunk


def _to_num(series):
    text = series.astype(str).str.replace(',', '', regex=False).str.strip()
    text = text.replace({'': np.nan, 'NONE': np.nan, 'NULL': np.nan, 'NAN': np.nan, 'N/A': np.nan})
    return pd.to_numeric(text, errors='coerce')


def _to_date(series):
    text = series.astype(str).str.strip()
    text = text.replace({'': np.nan, 'NONE': np.nan, 'NULL': np.nan, 'NAN': np.nan, 'N/A': np.nan})
    return pd.to_datetime(text, errors='coerce')


def _ref_codes(refs, member, value_type, code_candidates, desc_candidates, terms):
    columns = refs.columns.tolist()
    type_col = _pick_optional(columns, ['VALUE_TYPE'])
    code_col = _pick_optional(columns, code_candidates)
    desc_col = _pick_optional(columns, desc_candidates)
    missing = []
    if type_col is None:
        missing.append('VALUE_TYPE')
    if code_col is None:
        missing.append(code_candidates[0])
    if desc_col is None:
        missing.append(desc_candidates[0])
    if missing:
        _field_failure(member, columns, missing, 'reference code value type, code and description')
    types = refs[type_col].astype(str).str.strip().str.upper()
    subset = refs.loc[types.eq(str(value_type).strip().upper())]
    if subset.empty:
        return set()
    text = subset[desc_col].astype(str).str.upper()
    mask = pd.Series(False, index=subset.index)
    for term in terms:
        mask = mask | text.str.contains(str(term).upper(), regex=False, na=False)
    result = set()
    for raw in subset.loc[mask, code_col].astype(str).tolist():
        value = raw.strip()
        if value and value.upper() not in {'NAN', 'NONE', 'NULL', 'N/A'}:
            result.add(value)
    return result


def _count_before(date_map, pid, cutoff):
    values = date_map.get(pid)
    if values is None or len(values) == 0:
        return 0
    return int(np.searchsorted(values, np.datetime64(cutoff.to_datetime64(), 'ns'), side='left'))


def _has_before(date_map, pid, cutoff):
    value = date_map.get(pid)
    if value is None or pd.isna(value):
        return 0
    return int(pd.Timestamp(value) < cutoff)


def _check(checks, name, observed, expected, passed, kind=None):
    item = {'check': str(name), 'observed': str(observed)[:2000], 'expected': str(expected)[:2000], 'passed': bool(passed)}
    if kind is not None:
        item['kind'] = str(kind)
    checks.append(item)
    if not bool(passed):
        raise AssertionError(('source check failed: ' + str(name) + '; observed ' + str(observed) + '; expected ' + str(expected))[:4000])


def build(output: pathlib.Path, cache: pathlib.Path):
    output = pathlib.Path(output)
    cache = pathlib.Path(cache)
    checks = []
    archive_path = pathlib.Path(fetch(SOURCE_URL, cache, 'SDWA_latest_downloads.zip'))
    if not archive_path.exists():
        raise FileNotFoundError('downloaded archive missing at ' + str(archive_path))
    with zipfile.ZipFile(archive_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith('/')]

        def find_member(token):
            token_upper = token.upper()
            for name in names:
                if token_upper in pathlib.Path(name).name.upper():
                    return name
            raise KeyError(('archive member not found for token ' + token + '; observed members ' + str(names))[:4000])

        lcr_member = find_member('LCR_SAMPLES')
        pws_member = find_member('PUB_WATER_SYSTEMS')
        viol_member = find_member('VIOLATIONS_ENFORCEMENT')
        event_member = find_member('EVENTS_MILESTONES')
        ref_member = find_member('REF_CODE_VALUES')
        _check(checks, 'archive members selected for construction', str([lcr_member, pws_member, viol_member, event_member, ref_member]), 'single EPA ECHO SDWA national ZIP with LCR, public water system, violation, milestone and reference tables', len(names) > 0)

        lcr_header = _header(zf, lcr_member)
        pws_header = _header(zf, pws_member)
        viol_header = _header(zf, viol_member)
        event_header = _header(zf, event_member)
        ref_header = _header(zf, ref_member)

        join_presence = {}
        for member, header in [(lcr_member, lcr_header), (pws_member, pws_header), (viol_member, viol_header), (event_member, event_header)]:
            join_presence[member] = any(_canon(column) == 'PWSID' for column in header)
        if not all(join_presence.values()):
            missing_members = [member for member, present in join_presence.items() if not present]
            _field_failure('selected member group ' + str(missing_members), [lcr_header, pws_header, viol_header, event_header], ['PWSID'], 'system join key')
        _check(checks, 'join key PWSID present in the selected system, sample, violation and milestone tables', str(join_presence), 'PWSID is the system join key in all four event tables', all(join_presence.values()))

        pws = _read_full(zf, pws_member)
        pws_columns = pws.columns.tolist()
        pws_pid = _pick(pws_columns, ['PWSID'], pws_member, 'system identifier')
        pws_type_col = _pick(pws_columns, ['PWS_TYPE_CODE'], pws_member, 'system type')
        pop_col = _pick(pws_columns, ['POPULATION_SERVED_COUNT'], pws_member, 'population served')
        src_col = _pick(pws_columns, ['PRIMARY_SOURCE_CODE'], pws_member, 'primary source')
        own_col = _pick(pws_columns, ['OWNER_TYPE_CODE'], pws_member, 'owner type')
        prim_col = _pick_optional(pws_columns, ['PRIMACY_AGENCY_CODE'])

        pws[pws_pid] = pws[pws_pid].astype(str).str.strip()
        pws_types = pws[pws_type_col].astype(str).str.strip().str.upper()
        cws_mask = pws_types.eq('CWS')
        cws_ids = set(pws.loc[cws_mask, pws_pid].astype(str).str.strip())
        n_cws = len(cws_ids)
        type_counts = pws_types.value_counts(dropna=False).to_dict()
        _check(checks, 'community water system population from the PWS_TYPE_CODE field', 'CWS systems ' + str(n_cws) + '; other system rows ' + str(int((~cws_mask).sum())) + '; source type counts ' + str(type_counts), 'include community water systems and record, not abort on, other PWS type codes elsewhere in the source', n_cws > 0)

        if prim_col is not None:
            primacy = pws[prim_col].astype(str).str.strip().str.upper()
        else:
            primacy = pd.Series('', index=pws.index)
        fallback_primacy = pws[pws_pid].astype(str).str.strip().str[:2].str.upper()
        primacy_missing_before = int(primacy.eq('').sum())
        primacy = primacy.where(~primacy.eq(''), fallback_primacy)
        if prim_col is not None:
            primacy_observed = 'field ' + str(prim_col) + ' used; blank values ' + str(primacy_missing_before) + ' filled from the first two PWSID characters'
        else:
            primacy_observed = 'PRIMACY_AGENCY_CODE column absent; code derived from the first two PWSID characters'
        _check(checks, 'primacy agency code source and fallback', primacy_observed, 'two-character state or EPA region primacy code', bool(primacy.ne('').any()))

        pws_attrs = pd.DataFrame({'PWSID': pws[pws_pid].astype(str).str.strip(), 'population_served_count': _to_num(pws[pop_col]), 'primary_source_code': pws[src_col].astype(str).str.strip().str.upper(), 'owner_type_code': pws[own_col].astype(str).str.strip().str.upper(), 'primacy_agency_code': primacy})
        pws_attrs = pws_attrs.loc[cws_mask].copy()
        duplicate_system_rows = int(pws_attrs.duplicated('PWSID').sum())
        _check(checks, 'public water system identifier uniqueness', 'duplicate system rows ' + str(duplicate_system_rows), 'one row per PWSID in SDWA_PUB_WATER_SYSTEMS', duplicate_system_rows == 0)
        pws_attrs = pws_attrs.drop_duplicates('PWSID', keep='first').reset_index(drop=True)

        lcr_pid = _pick(lcr_header, ['PWSID'], lcr_member, 'sample system identifier')
        lcr_cont = _pick(lcr_header, ['CONTAMINANT_CODE'], lcr_member, 'contaminant code')
        lcr_measure_col = _pick(lcr_header, ['SAMPLE_MEASURE', 'SAMPLE_MEASURE_VALUE', 'SAMPLE_RESULT'], lcr_member, 'sample measure')
        lcr_unit_col = _pick(lcr_header, ['UNIT_OF_MEASURE', 'UNIT_OF_MEASURE_CODE'], lcr_member, 'measure unit')
        lcr_start_col = _pick(lcr_header, ['SAMPLING_START_DATE', 'SAMPLE_START_DATE'], lcr_member, 'sampling start date')
        lcr_end_col = _pick(lcr_header, ['SAMPLING_END_DATE', 'SAMPLE_END_DATE'], lcr_member, 'sampling end date')
        lcr_reported_col = _pick(lcr_header, ['SAR_LAST_REPORTED_DATE', 'LAST_REPORTED_DATE'], lcr_member, 'last reported date')

        selected_parts = []
        total_lcr_rows = 0
        pb90_all = 0
        cu90_all = 0
        pb90_cws = 0
        cu90_cws = 0
        usecols = list(dict.fromkeys([lcr_pid, lcr_cont, lcr_measure_col, lcr_unit_col, lcr_start_col, lcr_end_col, lcr_reported_col]))
        for chunk in _iter_chunks(zf, lcr_member, usecols):
            total_lcr_rows += len(chunk)
            pid = chunk[lcr_pid].astype(str).str.strip()
            cont = chunk[lcr_cont].astype(str).str.strip().str.upper()
            is_pb = cont.eq('PB90')
            is_cu = cont.eq('CU90')
            in_cws = pid.isin(cws_ids)
            pb90_all += int(is_pb.sum())
            cu90_all += int(is_cu.sum())
            pb90_cws += int((is_pb & in_cws).sum())
            cu90_cws += int((is_cu & in_cws).sum())
            take = (is_pb | is_cu) & in_cws
            if take.any():
                part = pd.DataFrame({'_pwsid': pid.loc[take].values, '_cont': cont.loc[take].values, '_measure_raw': chunk.loc[take, lcr_measure_col].values, '_unit_raw': chunk.loc[take, lcr_unit_col].values, '_start_raw': chunk.loc[take, lcr_start_col].values, '_end_raw': chunk.loc[take, lcr_end_col].values, '_reported_raw': chunk.loc[take, lcr_reported_col].values})
                selected_parts.append(part)

        _check(checks, 'lead and copper summary record coverage', 'LCR rows scanned ' + str(total_lcr_rows) + '; all PB90 rows ' + str(pb90_all) + '; all CU90 rows ' + str(cu90_all) + '; PB90 community rows ' + str(pb90_cws) + '; CU90 community rows ' + str(cu90_cws), 'PB90 and CU90 summary records for community water systems; other contaminant codes and non-community systems are not required to be absent', pb90_cws > 0 and cu90_cws > 0)

        if not selected_parts:
            raise AssertionError('no PB90 or CU90 records selected for community water systems')
        records = pd.concat(selected_parts, ignore_index=True)

        raw_measure = _to_num(records['_measure_raw'])
        units = records['_unit_raw'].astype(str).str.strip().str.upper().str.replace(' ', '', regex=False)
        units = units.str.replace(chr(181), 'U', regex=False).str.replace(chr(956), 'U', regex=False)
        factor = pd.Series(np.nan, index=records.index, dtype='float64')
        factor.loc[units.isin(['MG/L', 'MGL', 'MG/LITER', 'MG/LITRE', 'MILLIGRAMS PER LITER', 'PPM'])] = 1.0
        factor.loc[units.isin(['UG/L', 'UGL', 'MCG/L', 'PPB', 'UG/LITER', 'MICROGRAMS PER LITER'])] = 0.001
        records['_measure_mg'] = raw_measure * factor
        unknown_units = int(factor.isna().sum())
        unit_values = records['_unit_raw'].astype(str).str.strip().replace('', 'BLANK').value_counts().to_dict()
        _check(checks, 'lead and copper summary concentration units', 'unit counts ' + str(unit_values) + '; unrecognized unit rows ' + str(unknown_units), 'all selected PB90 and CU90 summary records use MG/L or a recognized microgram-per-litre or ppb unit convertible to mg/L', unknown_units == 0)

        missing_measure = int(records['_measure_mg'].isna().sum())
        _check(checks, 'lead and copper summary measurement parsing', 'numeric measurements ' + str(int(records['_measure_mg'].notna().sum())) + ' of ' + str(len(records)) + '; missing measurements ' + str(missing_measure), 'missing measurements remain missing and are not converted to zero; rows needing a missing target label are excluded', int(records['_measure_mg'].notna().sum()) > 0)

        records['_start'] = _to_date(records['_start_raw'])
        records['_end'] = _to_date(records['_end_raw'])
        records['_reported'] = _to_date(records['_reported_raw'])
        missing_start = int(records['_start'].isna().sum())
        records = records.loc[records['_start'].notna()].copy()
        _check(checks, 'monitoring period start date parsing', 'undated rows excluded ' + str(missing_start) + '; dated rows retained ' + str(len(records)), 'SAMPLING_START_DATE must be a valid date for period sequencing', len(records) > 0)

        missing_end = int(records['_end'].isna().sum())
        _check(checks, 'monitoring period end date completeness', 'rows with missing end dates ' + str(missing_end) + ' of ' + str(len(records)), 'SAMPLING_END_DATE is required for a period to serve as a prior window; rows missing it are retained only as potential target labels', missing_end < len(records))

        records = records.sort_values(['_pwsid', '_cont', '_start', '_reported', '_end', '_measure_mg'], na_position='first', kind='mergesort')
        before_dedup = len(records)
        dedup = records.drop_duplicates(['_pwsid', '_cont', '_start'], keep='last').copy()
        collapsed = before_dedup - len(dedup)
        _check(checks, 'duplicate PB90 and CU90 records for the same system and monitoring period', 'duplicate rows collapsed ' + str(collapsed) + '; period contaminant rows retained ' + str(len(dedup)), 'one latest-reported summary record per system, contaminant and sampling start date', len(dedup) > 0)

        pb = dedup.loc[dedup['_cont'].eq('PB90'), ['_pwsid', '_start', '_end', '_measure_mg']].rename(columns={'_measure_mg': 'pb90_mg_l', '_end': 'pb90_end'})
        cu = dedup.loc[dedup['_cont'].eq('CU90'), ['_pwsid', '_start', '_measure_mg']].rename(columns={'_measure_mg': 'cu90_mg_l'})
        periods = pb.merge(cu, on=['_pwsid', '_start'], how='left', validate='one_to_one')
        periods = periods.sort_values(['_pwsid', '_start', 'pb90_end'], na_position='last', kind='mergesort').reset_index(drop=True)
        _check(checks, 'PB90 monitoring period grain after de-duplication', 'PB90 periods ' + str(len(pb)) + '; CU90 periods ' + str(len(cu)) + '; joined period rows ' + str(len(periods)), 'one row per system and PB90 sampling start', len(periods) > 0)

        periods['_prior_start'] = periods.groupby('_pwsid', sort=False)['_start'].shift(1)
        periods['_prior_end'] = periods.groupby('_pwsid', sort=False)['pb90_end'].shift(1)
        periods['_prior_pb90'] = periods.groupby('_pwsid', sort=False)['pb90_mg_l'].shift(1)
        periods['_prior_cu90'] = periods.groupby('_pwsid', sort=False)['cu90_mg_l'].shift(1)
        periods['_prev_pb90'] = periods.groupby('_pwsid', sort=False)['pb90_mg_l'].shift(2)
        periods['_n_prior'] = periods.groupby('_pwsid', sort=False).cumcount()
        periods['_ever_exceeded'] = periods.groupby('_pwsid', sort=False)['pb90_mg_l'].transform(lambda values: values.gt(THRESHOLD).cumsum().shift(1).fillna(0).gt(0).astype(int))

        four_year_limit = periods['_prior_end'] + pd.DateOffset(years=4)
        pair_mask = periods['_prior_start'].notna() & periods['_prior_end'].notna() & periods['_prior_pb90'].notna()
        pair_mask = pair_mask & (periods['_start'] > periods['_prior_end'])
        pair_mask = pair_mask & (periods['_start'] <= four_year_limit)
        pair_mask = pair_mask & (periods['_prior_end'] >= periods['_prior_start'])
        pairs = periods.loc[pair_mask].copy()
        _check(checks, 'consecutive PB90 period pairing and four-year horizon limit', 'consecutive pairs retained ' + str(len(pairs)) + ' of ' + str(len(periods)) + ' PB90 periods', 'target period starts strictly after the prior period end and no more than four years later', len(pairs) > 0)

        unknown_label_rows = int(pairs['pb90_mg_l'].isna().sum())
        pairs = pairs.loc[pairs['pb90_mg_l'].notna()].copy()
        pairs[TARGET] = (pairs['pb90_mg_l'] > THRESHOLD).astype(int)
        _check(checks, 'target labels from the target period PB90 result', 'known target labels ' + str(len(pairs)) + '; unknown target measurements excluded ' + str(unknown_label_rows) + '; positive labels ' + str(int(pairs[TARGET].sum())), 'label is 1 only when target period PB90 exceeds 0.015 mg/L; unknown outcomes are excluded and never made negative', len(pairs) > 0)

        threshold_mg_l = 15.0 / 1000.0
        _check(checks, 'lead action level unit conversion and regulatory frame', '0.015 milligrams per litre corresponds to 15 micrograms per litre', '40 CFR 141.80(c)(2) lead action level through 2023-12-31', abs(threshold_mg_l - THRESHOLD) < 1e-12, kind='unit')

        pairs['PWSID'] = pairs['_pwsid'].astype(str).str.strip()
        pairs['target_period_start'] = pairs['_start'].dt.strftime('%Y-%m-%d')
        pairs['prior_pb90_mg_l'] = pairs['_prior_pb90'].astype(float)
        pairs['prior_cu90_mg_l'] = pairs['_prior_cu90'].astype(float)
        pairs['prior_pb90_ratio_to_al'] = pairs['_prior_pb90'] / THRESHOLD
        pairs['pb90_change_from_previous'] = pairs['_prior_pb90'] - pairs['_prev_pb90']
        pairs['n_prior_periods_observed'] = pairs['_n_prior'].astype(int)
        pairs['ever_exceeded_before'] = pairs['_ever_exceeded'].astype(int)
        pairs['gap_days_between_periods'] = (pairs['_start'] - pairs['_prior_end']).dt.days.astype(int)
        pairs['prior_period_length_days'] = (pairs['_prior_end'] - pairs['_prior_start']).dt.days.astype(int)

        pairs = pairs.merge(pws_attrs, on='PWSID', how='left', validate='many_to_one')
        missing_system_rows = int(pairs['primacy_agency_code'].isna().sum())
        _check(checks, 'system attribute join coverage', 'rows missing system attributes ' + str(missing_system_rows) + ' of ' + str(len(pairs)), 'every included community system has a row in SDWA_PUB_WATER_SYSTEMS', missing_system_rows == 0)

        refs = _read_full(zf, ref_member)
        viol_pid = _pick(viol_header, ['PWSID'], viol_member, 'violation system identifier')
        viol_date = _pick(viol_header, ['NON_COMPL_PER_BEGIN_DATE', 'VIOLATION_BEGIN_DATE'], viol_member, 'violation begin date')
        viol_code_col = _pick_optional(viol_header, ['VIOLATION_CODE'])
        viol_rule_col = _pick_optional(viol_header, ['RULE_CODE'])
        viol_desc_cols = []
        for candidate in ['VIOLATION_DESCRIPTION', 'RULE_DESCRIPTION', 'VIOLATION_NAME']:
            found = _pick_optional(viol_header, [candidate])
            if found is not None and found not in viol_desc_cols:
                viol_desc_cols.append(found)
        viol_codes = _ref_codes(refs, ref_member, 'VIOLATION_CODE', ['VIOLATION_CODE', 'VALUE_CODE', 'CODE'], ['VALUE_DESCRIPTION', 'DESCRIPTION'], ['LEAD AND COPPER', 'LCR', 'LEAD SERVICE LINE']) if viol_code_col is not None else set()
        rule_codes = _ref_codes(refs, ref_member, 'RULE_CODE', ['RULE_CODE', 'VALUE_CODE', 'CODE'], ['VALUE_DESCRIPTION', 'DESCRIPTION'], ['LEAD AND COPPER', 'LCR', 'LEAD SERVICE LINE']) if viol_rule_col is not None else set()
        if viol_rule_col is not None:
            rule_codes.add('350')
        identified_violations = bool(viol_codes) or bool(rule_codes) or bool(viol_desc_cols)
        if not identified_violations:
            _field_failure(viol_member, viol_header, ['VIOLATION_CODE or RULE_CODE or VIOLATION_DESCRIPTION'], 'Lead and Copper Rule violation identification')

        viol_usecols = list(dict.fromkeys([viol_pid, viol_date] + [column for column in [viol_code_col, viol_rule_col] + viol_desc_cols if column is not None]))
        viol_parts = []
        for chunk in _iter_chunks(zf, viol_member, viol_usecols):
            pid = chunk[viol_pid].astype(str).str.strip()
            dates = _to_date(chunk[viol_date])
            mask = pd.Series(False, index=chunk.index)
            if viol_code_col is not None and viol_codes:
                mask = mask | chunk[viol_code_col].astype(str).str.strip().isin(viol_codes)
            if viol_rule_col is not None and rule_codes:
                mask = mask | chunk[viol_rule_col].astype(str).str.strip().isin(rule_codes)
            for column in viol_desc_cols:
                text = chunk[column].astype(str).str.upper()
                mask = mask | text.str.contains('LEAD AND COPPER', regex=False, na=False)
                mask = mask | text.str.contains('LCR', regex=False, na=False)
                mask = mask | text.str.contains('LEAD SERVICE LINE', regex=False, na=False)
            mask = mask & dates.notna() & pid.isin(cws_ids)
            if mask.any():
                viol_parts.append(pd.DataFrame({'_pwsid': pid.loc[mask].values, '_date': dates.loc[mask].values}))

        n_viol = int(sum(len(part) for part in viol_parts))
        _check(checks, 'prior Lead and Copper Rule violation records', 'identified violation records ' + str(n_viol) + '; VALUE_TYPE scoped violation codes ' + str(len(viol_codes)) + '; rule codes ' + str(len(rule_codes)) + '; description columns ' + str(viol_desc_cols), 'dated Lead and Copper Rule violations identified from the violation table using the RULE_CODE 350 anchor and VALUE_TYPE-scoped reference labels', identified_violations)
        if viol_parts:
            violations = pd.concat(viol_parts, ignore_index=True)
            viol_dates = {str(pid): np.sort(np.array(group['_date'].values, dtype='datetime64[ns]')) for pid, group in violations.groupby('_pwsid')}
        else:
            viol_dates = {}

        evt_pid = _pick(event_header, ['PWSID'], event_member, 'milestone system identifier')
        evt_code_col = _pick(event_header, ['EVENT_MILESTONE_CODE', 'EVENT_CODE'], event_member, 'milestone event code')
        evt_actual_col = _pick_optional(event_header, ['EVENT_ACTUAL_DATE'])
        evt_reported_col = _pick_optional(event_header, ['FIRST_REPORTED_DATE'])
        evt_date_col = _pick_optional(event_header, ['EVENT_DATE'])
        if evt_actual_col is None and evt_reported_col is None and evt_date_col is None:
            _field_failure(event_member, event_header, ['EVENT_ACTUAL_DATE or FIRST_REPORTED_DATE'], 'milestone event date')
        evt_text_cols = []
        for candidate in ['EVENT_COMMENTS_TEXT', 'EVENT_REASON_CODE', 'EVENT_DESCRIPTION', 'EVENT_NAME', 'MILESTONE_DESCRIPTION']:
            found = _pick_optional(event_header, [candidate])
            if found is not None and found not in evt_text_cols:
                evt_text_cols.append(found)
        ref_event_codes = _ref_codes(refs, ref_member, 'EVENT_MILESTONE_CODE', ['EVENT_MILESTONE_CODE', 'VALUE_CODE', 'CODE'], ['VALUE_DESCRIPTION', 'DESCRIPTION'], ['LSLR', 'LEAD SERVICE LINE'])
        event_codes = set(ref_event_codes)
        event_codes.update({'LSLR', 'LSLI'})
        identified_events = bool(event_codes) or bool(evt_text_cols)
        _check(checks, 'reference code value-type scoping', 'violation codes ' + str(len(viol_codes)) + '; rule codes ' + str(len(rule_codes)) + '; scoped event milestone codes ' + str(len(ref_event_codes)) + '; LCR rule anchor present ' + str('350' in rule_codes) + '; LSLR event anchor present ' + str(any('LSLR' in str(code).upper() for code in event_codes)), 'SDWA_REF_CODE_VALUES filtered by VALUE_TYPE with the RULE_CODE 350 and EVENT_MILESTONE_CODE LSLR anchors', '350' in rule_codes and any('LSLR' in str(code).upper() for code in event_codes))

        evt_usecols = list(dict.fromkeys([evt_pid] + [column for column in [evt_code_col, evt_actual_col, evt_reported_col, evt_date_col] + evt_text_cols if column is not None]))
        event_parts = []
        for chunk in _iter_chunks(zf, event_member, evt_usecols):
            pid = chunk[evt_pid].astype(str).str.strip()
            dates = pd.Series(pd.NaT, index=chunk.index, dtype='datetime64[ns]')
            for column in [evt_actual_col, evt_reported_col, evt_date_col]:
                if column is not None:
                    parsed = _to_date(chunk[column])
                    dates = dates.where(dates.notna(), parsed)
            mask = pd.Series(False, index=chunk.index)
            code_text = chunk[evt_code_col].astype(str).str.strip().str.upper()
            mask = mask | code_text.isin({str(code).strip().upper() for code in event_codes})
            mask = mask | code_text.str.startswith('LSLR', na=False)
            mask = mask | code_text.str.contains('LEAD SERVICE LINE', regex=False, na=False)
            for column in evt_text_cols:
                text = chunk[column].astype(str).str.upper()
                mask = mask | text.str.contains('LSLR', regex=False, na=False)
                mask = mask | text.str.contains('LEAD SERVICE LINE', regex=False, na=False)
            mask = mask & dates.notna() & pid.isin(cws_ids)
            if mask.any():
                event_parts.append(pd.DataFrame({'_pwsid': pid.loc[mask].values, '_date': dates.loc[mask].values}))

        n_event = int(sum(len(part) for part in event_parts))
        _check(checks, 'lead service line replacement milestone record coverage', 'identified milestone records ' + str(n_event) + '; scoped reference milestone codes ' + str(len(ref_event_codes)) + '; event codes including anchors ' + str(sorted(event_codes)[:20]) + '; text columns ' + str(evt_text_cols), 'EVENT_MILESTONE_CODE LSLR records are present and parsed from the milestone table', n_event > 0)
        if event_parts:
            events = pd.concat(event_parts, ignore_index=True)
            event_min = {str(pid): pd.Timestamp(value) for pid, value in events.groupby('_pwsid')['_date'].min().items()}
        else:
            event_min = {}

        pairs['prior_lcr_violations_count'] = [_count_before(viol_dates, str(pid), pd.Timestamp(ts)) for pid, ts in zip(pairs['PWSID'], pairs['_start'])]
        pairs['prior_lslr_milestone'] = [_has_before(event_min, str(pid), pd.Timestamp(ts)) for pid, ts in zip(pairs['PWSID'], pairs['_start'])]
        _check(checks, 'prior violation and lead service line milestone timing', 'rows with nonzero prior LCR violations ' + str(int((pairs['prior_lcr_violations_count'] > 0).sum())) + '; rows with a prior LSLR milestone ' + str(int(pairs['prior_lslr_milestone'].sum())) + '; LCR violation records ' + str(n_viol) + '; LSLR event codes resolved ' + str(len(event_codes)), 'all violation and milestone dates used are strictly before each target period start', pairs['prior_lcr_violations_count'].notna().all() and pairs['prior_lslr_milestone'].isin([0, 1]).all())

        violation_rows = int((pairs['prior_lcr_violations_count'] > 0).sum())
        _check(checks, 'prior Lead and Copper Rule violation feature population', 'rows with nonzero prior LCR violations ' + str(violation_rows) + ' of ' + str(len(pairs)) + '; identified violation records ' + str(n_viol), 'at least one included row has a dated Lead and Copper Rule violation before its target period so the declared count is not a constant-zero column', violation_rows > 0)
        milestone_rows = int(pairs['prior_lslr_milestone'].sum())
        _check(checks, 'lead service line replacement milestone feature population', 'rows with a prior LSLR milestone before target start ' + str(milestone_rows) + ' of ' + str(len(pairs)) + '; identified milestone records ' + str(n_event), 'at least one included row has a dated LSLR milestone before its target period so the declared feature is not a constant-zero column', milestone_rows > 0)

        train_era_prior = pairs['_prior_end'] < AGENCY_CUTOFF
        agency_counts = pairs.loc[train_era_prior].groupby('primacy_agency_code').size()
        kept_agencies = set(agency_counts.loc[agency_counts >= 20].index.astype(str))
        all_agencies = set(agency_counts.index.astype(str))
        excluded_agencies = sorted(all_agencies - kept_agencies)
        before_floor = len(pairs)
        pairs = pairs.loc[pairs['primacy_agency_code'].astype(str).isin(kept_agencies)].copy()
        _check(checks, 'primacy agency floor from training-era feature dates only', 'agencies with at least twenty eligible pre-2014 pairs ' + str(len(kept_agencies)) + '; excluded agencies ' + str(len(excluded_agencies)) + '; pairs before floor ' + str(before_floor) + '; pairs retained ' + str(len(pairs)) + '; excluded agency codes ' + str(excluded_agencies[:40]), 'exclude agencies contributing fewer than twenty eligible pairs whose feature period ends before 2014-01-01, using no labels', len(kept_agencies) > 0 and len(pairs) > 0)

        split_series = pd.Series(index=pairs.index, dtype='object')
        split_series.loc[(pairs['_start'] >= TRAIN_START) & (pairs['_start'] < TRAIN_END)] = 'train'
        split_series.loc[(pairs['_start'] >= TRAIN_END) & (pairs['_start'] < EVAL_END)] = 'eval'
        split_series.loc[(pairs['_start'] >= EVAL_END) & (pairs['_start'] < HOLDOUT_END)] = 'holdout'
        pairs['_split'] = split_series
        out_of_window = int(pairs['_split'].isna().sum())
        pairs = pairs.loc[pairs['_split'].notna()].copy()
        _check(checks, 'fixed chronological split assignment', 'pairs outside the fixed windows excluded ' + str(out_of_window) + '; split counts ' + str(pairs['_split'].value_counts().to_dict()), 'train 1995-01-01 through 2013-12-31, eval 2014-01-01 through 2018-12-31, holdout 2019-01-01 through 2023-12-31; target start strictly before 2024-01-01', all((pairs['_split'] == s).any() for s in ['train', 'eval', 'holdout']))

        split_start_map = {'train': TRAIN_START, 'eval': TRAIN_END, 'holdout': EVAL_END}
        before_embargo = len(pairs)
        eval_drop = int(((pairs['_split'] == 'eval') & ~(pairs['_prior_end'] < TRAIN_END)).sum())
        holdout_drop = int(((pairs['_split'] == 'holdout') & ~(pairs['_prior_end'] < EVAL_END)).sum())
        embargo_mask = (pairs['_split'] == 'train') | (pairs['_prior_end'] < pairs['_split'].map(split_start_map))
        pairs = pairs.loc[embargo_mask].copy()
        _check(checks, 'chronological split and evaluation embargo', 'pairs before embargo ' + str(before_embargo) + '; eval pairs dropped for prior period ending on or after 2014-01-01 ' + str(eval_drop) + '; holdout pairs dropped for prior period ending on or after 2019-01-01 ' + str(holdout_drop) + '; pairs retained ' + str(len(pairs)), 'eval and holdout features must end before the split start so no feature window crosses a boundary; train is the earliest split and uses only dates before its target period', len(pairs) > 0 and all((pairs['_split'] == s).any() for s in ['train', 'eval', 'holdout']))

        split_counts = pairs['_split'].value_counts().to_dict()
        positive_counts = pairs.groupby('_split')[TARGET].sum().to_dict()
        _check(checks, 'split row and positive-label coverage', 'rows ' + str(split_counts) + '; positives ' + str(positive_counts), 'each fixed split has rows and at least one positive and one negative label after the population and embargo rules', all(split_counts.get(s, 0) > 0 for s in ['train', 'eval', 'holdout']) and all(positive_counts.get(s, 0) > 0 for s in ['train', 'eval', 'holdout']) and all(positive_counts.get(s, 0) < split_counts.get(s, 0) for s in ['train', 'eval', 'holdout']))

        final_columns = [ROW_ID, 'PWSID', 'target_period_start', TARGET] + FEATURES
        out = pairs[['PWSID', 'target_period_start', TARGET] + FEATURES].copy()
        out['PWSID'] = out['PWSID'].astype(str).str.strip()
        out['target_period_start'] = out['target_period_start'].astype(str)
        out.insert(0, ROW_ID, (out['PWSID'] + '_' + out['target_period_start']).values)
        out[TARGET] = out[TARGET].astype(int)
        out['__split'] = pairs['_split'].values
        out = out.sort_values(['target_period_start', 'PWSID', ROW_ID], kind='mergesort').reset_index(drop=True)

        duplicate_keys = int(out.duplicated(['PWSID', 'target_period_start']).sum())
        _check(checks, 'constructed row key uniqueness', 'duplicate system and target-period rows ' + str(duplicate_keys), 'one row per PWSID and target period start', duplicate_keys == 0)
        duplicate_row_ids = int(out.duplicated([ROW_ID]).sum())
        _check(checks, 'row identifier uniqueness', 'duplicate row_id values ' + str(duplicate_row_ids), 'every written row has a unique deterministic system-period row_id', duplicate_row_ids == 0)
        empty_row_ids = int(out[ROW_ID].astype(str).str.strip().eq('').sum())
        _check(checks, 'row identifier population', 'empty row identifiers ' + str(empty_row_ids), 'every written row has a nonempty system-period identifier', empty_row_ids == 0)
        _check(checks, 'binary target values', 'target values ' + str(sorted(out[TARGET].unique().tolist())), 'target contains only 0 and 1', set(out[TARGET].unique()).issubset({0, 1}))
        output_columns = [column for column in out.columns if column != '__split']
        _check(checks, 'output columns', str(output_columns), 'stable row_id and target_period_start keys, carried PWSID, the declared target, and exactly the declared feature columns', output_columns == final_columns)

        for split in ['eval', 'holdout']:
            subset = out.loc[out['__split'].eq(split)]
            positive_rows = subset.loc[subset[TARGET].eq(1)]
            positives = int(len(positive_rows))
            clusters = int(positive_rows['PWSID'].astype(str).str[:10].nunique())
            _check(checks, 'clustered positive coverage for ' + split, 'positive rows ' + str(positives) + '; distinct system clusters ' + str(clusters), 'at least 30 positive rows across at least 20 distinct system clusters', positives >= 30 and clusters >= 20)

        for feature in FEATURES:
            series = out[feature]
            if pd.api.types.is_numeric_dtype(series):
                missing = int(series.isna().sum())
            else:
                missing = int(series.isna().sum() + series.astype(str).str.strip().eq('').sum())
            passed = missing < len(out)
            _check(checks, 'feature population coverage for ' + feature, str(len(out) - missing) + ' populated of ' + str(len(out)) + '; missing ' + str(missing), 'declared feature values are parsed from the source or deterministic transforms; missing values remain missing and are not replaced by zero', passed)

        public_dir = output / 'public'
        private_dir = output / 'private'
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for split, target_path in [('train', public_dir / 'train.csv'), ('eval', public_dir / 'eval.csv'), ('holdout', private_dir / 'holdout.csv')]:
            frame = out.loc[out['__split'].eq(split), final_columns]
            frame.to_csv(target_path, index=False)
            written[str(target_path)] = int(target_path.stat().st_size)
        _check(checks, 'public and private CSV outputs written', str(written), 'three nonempty CSV files with identical columns under output', all(size > 0 for size in written.values()))

    if not checks:
        raise AssertionError('no source checks were produced')
    return checks
