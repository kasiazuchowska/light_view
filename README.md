# light_view – analiza widmowa źródeł światła

Aplikacja w Streamlit do definiowania widma źródła światła i wyznaczania jego parametrów
kolorymetrycznych: współrzędnych chromatyczności, temperatury barwowej najbliższej (CCT),
odległości od krzywej Plancka (Duv) oraz wskaźników oddawania barw (CIE CRI i ANSI/IES TM-30).
Widma są analizowane w zakresie 380–780 nm z krokiem 1 nm.

## Tryby definiowania widma

- **Rysuj** – odręczne narysowanie kształtu widma na płótnie (oś X: długość fali w nm,
  oś Y: natężenie względne 0–1).
- **Piki** – złożenie widma z pików gaussowskich; każdy pik definiuje się przez długość
  fali środka (nm), wysokość (0–1) i szerokość połówkową FWHM (nm).
- **Import** – wczytanie widma z pliku TXT. Każde widmo można też wyeksportować do takiego
  pliku przyciskiem „Eksportuj widmo (TXT)”.
- **Generuj** – wygenerowanie widma o zadanej CCT przez warunkowy autoenkoder wariacyjny
  (CVAE) wytrenowany na widmach rzeczywistych źródeł światła. Suwak „Różnorodność” skaluje
  wektor ukryty (0 = uśrednione widmo dla danej CCT), a „Nowa próbka” losuje nowy wektor.
  Zakres CCT danych treningowych to 1630–18273 K.

### Format pliku TXT

Dwie kolumny rozdzielone tabulatorem lub spacjami: długość fali w nm i natężenie.
Pierwszy wiersz może być nagłówkiem zaczynającym się od `wavelength` (jest pomijany).
Separatorem dziesiętnym jest **kropka**.

```
wavelength_nm	intensity
380	0.012345
381	0.013010
...
780	0.004100
```

Przy imporcie wartości spoza zakresu 380–780 nm są pomijane, dane są sortowane i
interpolowane liniowo do siatki 1 nm. Eksport zawsze zapisuje 401 punktów (380–780 nm co 1 nm).

## Wyznaczane parametry

- współrzędne chromatyczności **x, y** (CIE 1931, obserwator 2°) wraz z położeniem na
  wykresie chromatyczności,
- **CCT** metodą McCamy'ego (1992) i metodą Hernándeza-Andrésa (1999),
- **Duv** – odległość od krzywej Plancka w układzie CIE 1960 uv,
- **CIE CRI**: ogólny wskaźnik **Ra** oraz wskaźniki szczegółowe **R1–R14**,
- **ANSI/IES TM-30-18**: wskaźnik wierności **Rf**, wskaźnik gamutu **Rg**, wartości
  szczegółowe Rf dla 99 próbek CES oraz wykres wektorów barw.

## Instalacja

Wymagany jest [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Instalacja zależności (Python 3.12):

```bash
uv sync
```

## Uruchomienie

```bash
uv run streamlit run main.py
```

## Trening modelu

Wytrenowane wagi są zapisane w repozytorium (`model.pt`), więc aplikacja działa bez
ponownego treningu. Aby wytrenować model od nowa:

```bash
uv run python train.py
```

Dane są dzielone losowo (ziarno 42) na zbiór treningowy (85%) i testowy (15%); model jest
trenowany wyłącznie na zbiorze treningowym.

Funkcja straty składa się z trzech członów: błędu rekonstrukcji (MSE), dywergencji KL
z wagą β oraz członu zgodności CCT z wagą λ_CCT. Człon CCT dotyczy widm generowanych
z rozkładu a priori dla losowych zadanych CCT: ich CCT jest liczona różniczkowalnie
(funkcje dopasowania barw CIE 1931 2°, wzór McCamy'ego) i porównywana z zadaną (MSE
w znormalizowanej skali logarytmicznej). Obie wagi rosną liniowo od 0 przez pierwsze
100 epok, do β = 0,01 i λ_CCT = 1.

Skrypt tworzy:

- `model.pt` – wagi wytrenowanego modelu,
- `training_log.csv` – przebieg treningu: dla każdej epoki wagi β i λ_CCT, średnia strata
  całkowita, rekonstrukcji i KL na zbiorze treningowym i testowym oraz średni człon CCT
  na zbiorze treningowym,
- `data/test_indices.json` – indeksy widm należących do zbioru testowego (względem
  kolejności zwracanej przez `load_dataset()`).

## Pliki projektu

- `main.py` – interfejs Streamlit: wybór trybu, wyświetlanie wyników, import i eksport TXT,
- `spectrum.py` – obliczenia kolorymetryczne (xy, Duv, CRI, TM-30), widmo z pików gaussowskich,
- `diagrams.py` – wykresy Plotly: widmo, wykres chromatyczności CIE 1931, R1–R14, Rf
  szczegółowe, wykres wektorów barw TM-30,
- `canvas.py` – tło płótna do rysowania i odczyt widma z narysowanej krzywej,
- `model.py` – architektura warunkowego VAE i normalizacja CCT,
- `data_loader.py` – wczytanie widm i CCT z pliku Excel oraz przepróbkowanie do 380–780 nm co 1 nm,
- `train.py` – trening modelu,
- `data/real_light_source_spds.xlsx` – dane treningowe.

## Źródło danych treningowych

*Real Light Source SPDs For Use in Research*, figshare, pozycja 12947240.
