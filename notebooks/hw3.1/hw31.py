import json
import re
import argparse
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd


BUREAU_COLUMNS = [
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "CREDIT_CURRENCY",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE",
    "CREDIT_TYPE",
    "DAYS_CREDIT_UPDATE",
    "AMT_ANNUITY",
]

POS_CASH_BALANCE_COLUMNS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]


def convert_scalar(value: Any) -> Any:
    """
    Преобразует значение в int, float или None, если это возможно.
    Иначе возвращает исходное значение.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        if value == "None":
            return None

        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value

    return value


def parse_embedded_object(raw_value: str) -> Dict[str, Any]:
    """
    Парсит строку вида:
    SomeObject(field1=value1, field2='text', field3=None)
    и возвращает словарь полей.
    """
    result: Dict[str, Any] = {}

    if not raw_value:
        return result

    match = re.search(r"\w+\((.*)\)$", raw_value)
    if not match:
        return result

    inner_part = match.group(1)
    pattern = r"(\w+)=('.*?'|None|-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, inner_part)

    for key, value in matches:
        cleaned_value = value.strip("'")
        result[key] = convert_scalar(cleaned_value)

    return result


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Парсит строку .log как JSON.
    Возвращает None для пустых или битых строк.
    """
    line = line.strip()
    if not line:
        return None

    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def parse_bureau_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Преобразует уже распарсенный payload типа bureau
    в одну строку будущего bureau.csv.
    """
    if payload.get("type") != "bureau":
        return None

    data = payload.get("data", {})
    record = data.get("record", {})
    amt_credit_data = parse_embedded_object(record.get("AmtCredit", ""))

    return {
        "SK_ID_CURR": record.get("SK_ID_CURR"),
        "SK_ID_BUREAU": record.get("SK_ID_BUREAU"),
        "CREDIT_ACTIVE": record.get("CREDIT_ACTIVE"),
        "CREDIT_CURRENCY": amt_credit_data.get("CREDIT_CURRENCY"),
        "DAYS_CREDIT": record.get("DAYS_CREDIT"),
        "CREDIT_DAY_OVERDUE": record.get("CREDIT_DAY_OVERDUE"),
        "DAYS_CREDIT_ENDDATE": record.get("DAYS_CREDIT_ENDDATE"),
        "DAYS_ENDDATE_FACT": record.get("DAYS_ENDDATE_FACT"),
        "AMT_CREDIT_MAX_OVERDUE": amt_credit_data.get("AMT_CREDIT_MAX_OVERDUE"),
        "CNT_CREDIT_PROLONG": record.get("CNT_CREDIT_PROLONG"),
        "AMT_CREDIT_SUM": amt_credit_data.get("AMT_CREDIT_SUM"),
        "AMT_CREDIT_SUM_DEBT": amt_credit_data.get("AMT_CREDIT_SUM_DEBT"),
        "AMT_CREDIT_SUM_LIMIT": amt_credit_data.get("AMT_CREDIT_SUM_LIMIT"),
        "AMT_CREDIT_SUM_OVERDUE": amt_credit_data.get("AMT_CREDIT_SUM_OVERDUE"),
        "CREDIT_TYPE": data.get("CREDIT_TYPE"),
        "DAYS_CREDIT_UPDATE": record.get("DAYS_CREDIT_UPDATE"),
        "AMT_ANNUITY": amt_credit_data.get("AMT_ANNUITY"),
    }


def parse_pos_cash_balance_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Преобразует уже распарсенный payload типа POS_CASH_balance
    в список строк будущего POS_CASH_balance.csv.
    """
    if payload.get("type") != "POS_CASH_balance":
        return []

    data = payload.get("data", {})
    cnt_instalment = data.get("CNT_INSTALMENT")
    records = data.get("records", [])

    rows: List[Dict[str, Any]] = []

    for record in records:
        ids_data = parse_embedded_object(record.get("PosCashBalanceIDs", ""))

        rows.append({
            "SK_ID_PREV": ids_data.get("SK_ID_PREV"),
            "SK_ID_CURR": ids_data.get("SK_ID_CURR"),
            "MONTHS_BALANCE": record.get("MONTHS_BALANCE"),
            "CNT_INSTALMENT": cnt_instalment,
            "CNT_INSTALMENT_FUTURE": record.get("CNT_INSTALMENT_FUTURE"),
            "NAME_CONTRACT_STATUS": ids_data.get("NAME_CONTRACT_STATUS"),
            "SK_DPD": record.get("SK_DPD"),
            "SK_DPD_DEF": record.get("SK_DPD_DEF"),
        })

    return rows


def process_log_file(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Обрабатывает .log файл в один проход и возвращает:
    - строки POS_CASH_balance
    - строки bureau
    """
    pos_cash_rows: List[Dict[str, Any]] = []
    bureau_rows: List[Dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            payload = parse_log_line(line)
            if not payload:
                continue

            record_type = payload.get("type")

            if record_type == "POS_CASH_balance":
                pos_cash_rows.extend(parse_pos_cash_balance_payload(payload))
            elif record_type == "bureau":
                bureau_row = parse_bureau_payload(payload)
                if bureau_row is not None:
                    bureau_rows.append(bureau_row)

    return pos_cash_rows, bureau_rows


def save_dataframe(rows: List[Dict[str, Any]], columns: List[str], output_path: str) -> None:
    """
    Сохраняет список словарей в CSV через DataFrame.
    """
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    """
    Полный цикл:
    1. Поточное чтение входного .log файла
    2. Обработка обеих сущностей в один проход
    3. Сохранение результатов в два .csv файла
    """
    parser = argparse.ArgumentParser(
        description="Парсинг .log файла в POS_CASH_balance.csv и bureau.csv"
    )
    parser.add_argument("input_log", help="Путь к входному .log файлу")
    parser.add_argument("pos_cash_output", help="Путь к выходному POS_CASH_balance.csv")
    parser.add_argument("bureau_output", help="Путь к выходному bureau.csv")

    args = parser.parse_args()

    pos_cash_rows, bureau_rows = process_log_file(args.input_log)

    save_dataframe(pos_cash_rows, POS_CASH_BALANCE_COLUMNS, args.pos_cash_output)
    save_dataframe(bureau_rows, BUREAU_COLUMNS, args.bureau_output)

    print(f"Сохранён POS_CASH_balance: {args.pos_cash_output}")
    print(f"Сохранён bureau: {args.bureau_output}")
    print(f"Строк POS_CASH_balance: {len(pos_cash_rows)}")
    print(f"Строк bureau: {len(bureau_rows)}")


if __name__ == "__main__":
    main()