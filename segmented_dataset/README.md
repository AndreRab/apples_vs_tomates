# Segmentacja całego datasetu

Metoda: Lab + Otsu z notebooka task3.ipynb, morfologia i największy komponent.
Parametry identyczne dla train/test, bez dostrajania na teście.

- `images/`: obrazy PNG RGBA z przezroczystym tłem, w oryginalnym rozmiarze.
- `masks/`: maski PNG, 255 = obiekt, 0 = tło.
- W obu folderach zachowano podział train/test i apples/tomatoes.
- Nazwa `oryginal.jpeg.png` zachowuje pełną nazwę pliku źródłowego.
- `manifest.csv`: mapowanie źródeł na wyniki oraz diagnostyka.

Przetworzono 391 obrazów. Heurystyka oznaczyła 56 masek do przeglądu.
To segmentacja wstępna: może wybierać tło, pomijać owoce lub zachowywać tylko jeden
z wielu rozłącznych obiektów. Ostrzeżenia nie wykrywają wszystkich błędów.
Duplikaty i oryginalny podział zbioru zostały zachowane.

Ponowne generowanie: `uv run python segment_dataset.py` z folderu third_task.
