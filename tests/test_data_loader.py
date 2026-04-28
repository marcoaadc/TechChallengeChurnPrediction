"""Testes unitários para o módulo src.data_loader."""

import pandas as pd
import pytest

from src.data_loader import load_data


class TestLoadData:
    """Testes da função load_data."""

    def test_load_valid_csv(self, tmp_path):
        """Carrega um CSV válido e retorna DataFrame com dados corretos."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col_a,col_b,col_c\n1,2,3\n4,5,6\n")
        df = load_data(str(csv_file))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["col_a", "col_b", "col_c"]

    def test_file_not_found_raises(self):
        """FileNotFoundError ao tentar carregar arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            load_data("/caminho/que/nao/existe/dados.csv")

    def test_empty_csv_returns_empty_dataframe(self, tmp_path):
        """CSV só com header retorna DataFrame vazio (0 linhas)."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("col_a,col_b\n")
        df = load_data(str(csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["col_a", "col_b"]

    def test_csv_with_special_characters(self, tmp_path):
        """CSV com caracteres especiais (acentos) é carregado corretamente."""
        csv_file = tmp_path / "special.csv"
        csv_file.write_text("nome,descrição\nJoão,ação\nMaria,opção\n", encoding="utf-8")
        df = load_data(str(csv_file))
        assert len(df) == 2
        assert df.iloc[0]["nome"] == "João"

    def test_numeric_columns_parsed_correctly(self, tmp_path):
        """Colunas numéricas são parseadas com dtypes corretos."""
        csv_file = tmp_path / "numeric.csv"
        csv_file.write_text("int_col,float_col,str_col\n1,1.5,abc\n2,2.5,def\n")
        df = load_data(str(csv_file))
        assert pd.api.types.is_integer_dtype(df["int_col"])
        assert pd.api.types.is_float_dtype(df["float_col"])
        assert pd.api.types.is_object_dtype(df["str_col"])
