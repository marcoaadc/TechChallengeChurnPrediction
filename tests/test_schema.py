"""Testes de validação de schema do dataset com pandera."""

import pandas as pd
import pandera.pandas as pa

telco_schema = pa.DataFrameSchema(
    {
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": pa.Column(int, pa.Check.ge(0)),
        "PhoneService": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "Contract": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
        "TotalCharges": pa.Column(float, pa.Check.ge(0)),
        "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
    },
    coerce=True,
)


def _load_clean_dataset() -> pd.DataFrame:
    """Carrega e limpa o dataset para validação."""
    from src.data_loader import load_telco_churn

    df = load_telco_churn()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    return df


def test_dataset_schema_validation():
    """Valida que o dataset carregado segue o schema esperado."""
    df = _load_clean_dataset()
    validated = telco_schema.validate(df)
    assert len(validated) > 5000


def test_dataset_no_nulls_after_cleaning():
    """Após limpeza, o dataset não deve ter valores nulos."""
    df = _load_clean_dataset()
    assert df.isnull().sum().sum() == 0


def test_dataset_churn_balance():
    """A classe positiva (Churn=Yes) deve estar entre 20% e 35%."""
    df = _load_clean_dataset()
    churn_rate = (df["Churn"] == "Yes").mean()
    assert 0.20 <= churn_rate <= 0.35
