# Task 3 — analiza i segmentacja

Samodzielny projekt uv. Instalacja i uruchomienie:

```sh
cd third_task
uv sync --locked
uv run python -m ipykernel install --user --name third-task --display-name "Python (third_task uv)"
```

Otwórz `task3.ipynb` w edytorze i wybierz kernel `Python (third_task uv)` lub interpreter `third_task/.venv/bin/python`. Notebook ma zapisane wyniki wykonania wszystkich komórek.

`outputs/` zawiera inwentaryzację CSV, diagnostykę masek treningowych i siatki PNG z przykładami. Dane źródłowe pozostają bez zmian. Wykryto identyczne obrazy między train/test; szczegóły i ograniczenia segmentacji opisano w notebooku.
