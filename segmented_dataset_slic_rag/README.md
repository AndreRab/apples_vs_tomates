Segmentacja SLIC + RAG z sekcji 8 task3.ipynb.
images/: PNG RGBA; masks/: maski 0/255; manifest.csv: diagnostyka i źródła.
Parametry: 200 superpikseli, compactness=10, sigma=1, próg scalania RAG=15 w Lab.
Wybór obiektu: odległość koloru regionu od tła + Otsu, filtr kontaktu z brzegiem i małych składowych.
Puste maski pozostają puste. Brak gwarancji semantycznej poprawności.
