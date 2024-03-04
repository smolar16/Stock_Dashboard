import base64
from flask import Flask, render_template, request, redirect, url_for
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import numpy as np
import datetime as dt

app = Flask(__name__)


def fetch_and_process_data():
    # Lista wskaźników rentowności i indeksów giełdowych
    ratios = ["ROE", "ROA", "OPM", "ROS", "RS", "GPM", "RBS", "ROPA"]
    stock_indices = ["WIG20", "mWIG40", "sWIG80"]

    # Bazowy URL do pobierania danych
    base_url = "https://www.biznesradar.pl/spolki-wskazniki-rentownosci/indeks:{},{},2,2"

    dataframes = {}  # Słownik przechowujący ramki danych dla różnych indeksów i wskaźników

    for stock_index in stock_indices:
        for ratio in ratios:
            # Tworzenie URL na podstawie indeksu giełdowego i wskaźnika
            url = base_url.format(stock_index, ratio)
            # Wysłanie żądania HTTP, aby pobrać stronę internetową
            response = requests.get(url)

            if response.status_code == 200:
                # Parsowanie strony HTML przy użyciu biblioteki BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                # Znalezienie tabeli zawierającej dane
                table = soup.find('table', class_='qTableFull')

                data = []  # Lista przechowująca dane wierszy
                headers = []  # Lista przechowująca nagłówki kolumn
                for row in table.find_all('tr'):
                    # Znajdź wszystkie elementy 'td' (komórki danych) i 'th' (nagłówki kolumn) w bieżącym wierszu
                    cols = row.find_all(['td', 'th'])
                    # Inicjalizacja listy na dane w bieżącym wierszu
                    row_data = [ele.text.strip() for ele in cols]

                    if row.find('th'):
                        # Jeśli bieżący wiersz zawiera nagłówki kolumn
                        headers = row_data  # Przypisz zawartość bieżącego wiersza do zmiennej 'headers'
                    else:
                        # Jeśli bieżący wiersz zawiera dane
                        data.append(row_data)  # Dodaj dane z bieżącego wiersza do listy 'data'

                # Tworzenie data frame z pobranych danych
                df = pd.DataFrame(data, columns=headers)
                # Usunięcie niepotrzebnych kolumn z data frame
                df = df.drop(df.columns[3:9], axis=1)

                # Tworzenie unikalnego klucza dla danej kombinacji indeksu i wskaźnika
                df_key = f"{stock_index}_{ratio}"
                dataframes[df_key] = df
            else:
                # W przypadku nieudanego pobrania danych, wyświetlenie komunikatu o błędzie
                print(f"Nie udało się pobrać danych dla {stock_index}, {ratio}")

    # Łączenie danych dla różnych indeksów
    wig20_df = merge_dataframes("WIG20", dataframes, ratios)
    mwig40_df = merge_dataframes("mWIG40", dataframes, ratios)
    swig80_df = merge_dataframes("sWIG80", dataframes, ratios)

    # Usunięcie wiersza z indeksem 30 z DataFrame swig80_df
    swig80_df = swig80_df.drop(swig80_df.index[30])

    # Sortowanie każdej ramki danych względem zerowej kolumny
    wig20_df = wig20_df.sort_values(by=wig20_df.columns[0])
    mwig40_df = mwig40_df.sort_values(by=mwig40_df.columns[0])
    swig80_df = swig80_df.sort_values(by=swig80_df.columns[0])

    try:
        # Ustal ścieżkę, w której mają być zapisane pliki CSV
        current_dir = os.getcwd()
        wig20_df.to_csv(os.path.join(current_dir, 'wig20_df.csv'), index=False)
        mwig40_df.to_csv(os.path.join(current_dir, 'mwig40_df.csv'), index=False)
        swig80_df.to_csv(os.path.join(current_dir, 'swig80_df.csv'), index=False)
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisywania danych: {e}")

    return wig20_df, mwig40_df, swig80_df


def merge_dataframes(index, dataframes, indicators):
    # Inicjalizacja zmiennej 'merged_df' jako pustego DataFrame
    merged_df = None

    for indicator in indicators:
        # Tworzenie klucza do identyfikacji danego DataFrame
        df_key = f"{index}_{indicator}"

        if df_key in dataframes:
            # Jeśli DataFrame dla danego indeksu i wskaźnika istnieje w słowniku
            if merged_df is None:
                # Jeśli 'merged_df' jest pusty (pierwsza iteracja), przypisz jej wartość DataFrame
                merged_df = dataframes[df_key]
            else:
                # W przeciwnym razie, łącz 'merged_df' z nowy DataFrame w oparciu o kolumnę pierwszą
                merged_df = pd.merge(merged_df,
                                     dataframes[df_key][[dataframes[df_key].columns[0], dataframes[df_key].columns[2]]],
                                     on=dataframes[df_key].columns[0], how='left')

    # Zwrócenie połączonego DataFrame
    return merged_df


def read_csv_data():
    try:
        # Próba odczytu danych z plików CSV
        wig20_df = pd.read_csv('wig20_df.csv')
        mwig40_df = pd.read_csv('mwig40_df.csv')
        swig80_df = pd.read_csv('swig80_df.csv')

        # Zwrócenie odczytanych ramek danych
        return wig20_df, mwig40_df, swig80_df
    except FileNotFoundError:
        # W przypadku braku plików CSV, wyświetlenie komunikatu o błędzie i zwrócenie None dla każdej ramki danych
        print("Nie znaleziono plików CSV. Wykonaj najpierw 'fetch_and_process_data()'.")
        return None, None, None

def get_data(stocks, start, end):
    try:
        # Pobiera dane giełdowe dla podanych akcji z serwisu 'stooq' za określony okres czasu.
        stockData = web.DataReader(stocks, 'stooq', start, end)

        # Wybiera kolumnę 'Close', która reprezentuje ceny zamknięcia akcji.
        stockData = stockData['Close']

        # Oblicza procentową zmianę cen zamknięcia, co jest podstawą do obliczenia zwrotów.
        returns = stockData.pct_change()

        # Oblicza średni zwrot dla każdej akcji.
        meanReturns = returns.mean()

        # Oblicza macierz kowariancji zwrotów, która jest używana do oceny ryzyka portfela.
        covMatrix = returns.cov()

        return meanReturns, covMatrix
    except KeyError:
        # Wyjątek zgłaszany, gdy nie można znaleźć danych dla podanych symboli akcji.
        raise ValueError("Jeden lub więcej tickerów akcji jest nieprawidłowe lub dane są niedostępne..")


def monte_carlo_simulation(stocks, start_date, end_date, num_sims, time_horizon, initial_investment, weights):
    # Pobiera średnie zwroty i macierz kowariancji dla podanych akcji.
    meanReturns, covMatrix = get_data(stocks, start_date, end_date)

    # Ustawienie liczby symulacji oraz horyzontu czasowego symulacji.
    mc_sims = num_sims
    T = time_horizon

    # Tworzy macierz pełną średnich zwrotów dla każdego dnia symulacji.
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanM = meanM.T

    # Inicjalizuje macierz, która będzie przechowywać wyniki symulacji portfela.
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    # Pętla wykonująca symulacje Monte Carlo.
    for m in range(0, mc_sims):
        # Generuje niezależne zmienne losowe dla symulacji.
        Z = np.random.normal(size=(T, len(weights)))

        # Dekompozycja Cholesky'ego macierzy kowariancji do generowania skorelowanych zwrotów.
        L = np.linalg.cholesky(covMatrix)

        # Obliczanie skorelowanych dziennych zwrotów dla akcji.
        dailyReturns = meanM + np.inner(L, Z)

        # Oblicza wartość portfela dla każdego dnia symulacji.
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initial_investment

    # Tworzy wykres z wyników symulacji.
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_sims)
    plt.ylabel('Wartość Portfela (PLN)')
    plt.xlabel('Dni')
    plt.title('Symulacja Monte Carlo dla Portfela Akcji')

    # Usuwa marginesy z osi X na wykresie.
    plt.margins(x=0)

    # Zapisuje wykres do pamięci, a nie do pliku.
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)

    # Koduje wykres do formatu Base64, aby można go było wyświetlić na stronie internetowej.
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Oblicza przewidywaną wartość portfela na koniec symulacji.
    portfolio_last = portfolio_sims[-1]
    expected_value = round(np.mean(portfolio_last), 2)

    # Oblicza stopę zwrotu portfela na podstawie wartości początkowej i końcowej.
    return_rate = ((expected_value - initial_investment) / initial_investment) * 100

    return plot_url, expected_value, return_rate



###########################################

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/monte_carlo', methods=['GET', 'POST'])
def monte_carlo():
    # Obsługuje żądania POST, gdy formularz na stronie jest przesyłany
    if request.method == 'POST':
        # Pobiera nazwy akcji i wagi z formularza i dzieli je na listy
        stocks = request.form.get('stocks').split(',')
        weights_input = request.form.get('weights').split(',')

        try:
            # Próbuje przekonwertować wagi na liczby zmiennoprzecinkowe
            weights = [float(w) for w in weights_input]

            # Sprawdza, czy suma wag równa się 1, jeśli nie - zwraca błąd
            if sum(weights) != 1.0:
                return render_template('monte_carlo.html', error="Suma wag musi wynosić 1.")
        except ValueError:
            # Jeśli konwersja wag na liczby zmiennoprzecinkowe zawiedzie, zwraca błąd
            return render_template('monte_carlo.html', error="Nieprawidłowy format wagi. Wprowadź prawidłowe liczby.")

        # Konwertuje pozostałe dane formularza na odpowiednie formaty
        start_date = dt.datetime.strptime(request.form.get('start_date'), '%Y-%m-%d')
        end_date = dt.datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')
        num_sims = int(request.form.get('num_sims'))
        time_horizon = int(request.form.get('time_horizon'))
        initial_investment = float(request.form.get('initial_investment'))

        try:
            # Wywołuje funkcję symulacji Monte Carlo i przechwytuje wyniki
            plot_url, expected_value, return_rate = monte_carlo_simulation(
                stocks, start_date, end_date, num_sims, time_horizon, initial_investment, weights
            )
        except ValueError as e:
            # Przechwytuje wyjątek, jeśli wystąpi błąd w symulacji i wyświetla komunikat o błędzie
            return render_template('monte_carlo.html', error=str(e))

        # Formatuje wagi do wyświetlenia w szablonie HTML
        weights_display = ["{:.2%}".format(weight) for weight in weights]
        # Tworzy słownik połączonych nazw akcji i ich wag do wyświetlenia
        stocks_weights = [{"stock": stock, "weight": weight} for stock, weight in zip(stocks, weights_display)]

        # Zwraca szablon HTML z wynikami symulacji
        return render_template('monte_carlo.html', plot_url=plot_url, expected_value=expected_value,
                               return_rate=return_rate, stocks_weights=stocks_weights)

    # Jeśli żądanie nie jest typu POST, zwraca pusty szablon monte_carlo.html
    return render_template('monte_carlo.html', plot_url=None, expected_value=None, return_rate=None,
                           stocks_weights=None)


@app.route('/chart/<ticker>', methods=['GET', 'POST'])
def chart(ticker, start_date=None, end_date=None):
    if request.method == 'POST':
        # Jeśli jest to żądanie POST (np. formularz został przesłany)
        # Ustaw zakres dat na podstawie danych z formularza
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
    else:
        # Jeśli to nie jest żądanie POST, ustaw domyślne zakresy dat, jeśli nie zostały podane
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

    # Pobieranie danych giełdowych za pomocą pandas_datareader
    try:
        df = web.DataReader(ticker, 'stooq', start=start_date, end=end_date)
    except Exception as e:
        return render_template('index.html', error_message="Błąd: Nie można pobrać danych dla podanego zakresu dat.")

    # Sprawdzenie, czy DataFrame nie jest pusty
    if df.empty:
        return render_template('index.html', error_message="Błąd: Brak danych dla podanego zakresu dat lub symbolu spółki.")

    # Obliczanie stopy zwrotu
    if not df.empty:
        start_price = df['Close'].iloc[-1]
        end_price = df['Close'].iloc[0]
        return_rate = ((end_price - start_price) / start_price) * 100
        return_rate_text = f"({end_price} - {start_price}) / {start_price} * 100"
    else:
        return_rate = 0
        return_rate_text = "Brak danych"

    # Generowanie wykresu
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label=f'{ticker} Close')
    ax.set_title(f'Wykres {ticker}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Cena')
    ax.legend()
    ax.grid(True)

    # Ustawienie zakresu osi X
    if not df.empty:
        ax.set_xlim(df.index.min(), df.index.max())

    # Zapisywanie wykresu do obiektu bytes w pamięci, zamiast do pliku
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close(fig)
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    # Przekazanie zakodowanego obrazu do szablonu HTML
    return render_template('index.html', img_data=img_base64, ticker=ticker,
                           return_rate=return_rate, start_date=start_date, end_date=end_date,
                           return_rate_text=return_rate_text)

# Funkcja pobierające dane z web scrapingu
@app.route('/update_data')
def update_data():
    # Pobierz dane i zapisz je do plików CSV za pomocą funkcji 'fetch_and_process_data'
    fetch_and_process_data()

    # Przekazujemy DataFrame i informację o ścieżce zapisu do szablonu 'update.html'
    return render_template('update.html',
                           title='Aktualizacja Danych',
                           )


# Funkcja obsługująca endpoint '/wig20'
@app.route('/wig20', methods=['GET', 'POST'])
def wig20():
        return chart('WIG20.PL')

# Funkcja obsługująca endpoint '/mwig40'
@app.route('/mwig40', methods=['GET', 'POST'])
def mwig40():

        return chart('mWIG40.PL')

# Funkcja obsługująca endpoint '/swig80'
@app.route('/swig80', methods=['GET', 'POST'])
def swig80():
        return chart('sWIG80.PL')

@app.route('/company_chart', methods=['GET', 'POST'])
def company_chart():
    if request.method == 'POST':
        # Jeśli to żądanie POST (np. formularz został przesłany)
        # Pobierz symbol spółki z formularza
        symbol = request.form.get('symbol')
        try:
            # Przekieruj do funkcji generującej wykres (endpoint 'chart')
            return redirect(url_for('chart', ticker=symbol))
        except Exception as e:
            # Obsługa błędu, jeśli symbol jest niepoprawny
            return render_template('company_chart.html', error_message=f"Błąd: Wprowadzono niepoprawny symbol. {e}")

    # Jeśli to nie jest żądanie POST, wyświetl formularz na stronie 'company_chart.html'
    return render_template('company_chart.html', error_message=None)


# Funkcja obsługująca endpoint '/component/wig20'
@app.route('/component/wig20')
def component_wig20():
    # Odczytanie danych z plików CSV za pomocą funkcji 'read_csv_data'
    wig20_df, _, _ = read_csv_data()

    if wig20_df is not None:
        # Jeśli ramka danych 'wig20_df' nie jest pusta
        # Wygeneruj szablon HTML 'component.html' z danymi z 'wig20_df'
        return render_template('component.html', data_html=wig20_df.to_html(classes='data', index=False),
                               title='Komponenty WIG 20')
    else:
        # Jeśli dane nie zostały jeszcze załadowane, wyświetl komunikat o błędzie
        return render_template('component.html', message='Dane nie zostały załadowane. Dokonaj aktualizacji danych.')


# Funkcja obsługująca endpoint '/component/mwig40'
@app.route('/component/mwig40')
def component_mwig40():
    # Odczytanie danych z plików CSV za pomocą funkcji 'read_csv_data'
    _, mwig40_df, _ = read_csv_data()

    if mwig40_df is not None:
        # Jeśli ramka danych 'mwig40_df' nie jest pusta
        # Wygeneruj szablon HTML 'component.html' z danymi z 'mwig40_df'
        return render_template('component.html', data_html=mwig40_df.to_html(classes='data', index=False),
                               title='Komponenty mWIG 40')
    else:
        # Jeśli dane nie zostały jeszcze załadowane, wyświetl komunikat o błędzie
        return render_template('component.html', message='Dane nie zostały załadowane. Dokonaj aktualizacji danych.')


# Funkcja obsługująca endpoint '/component/swig80'
@app.route('/component/swig80')
def component_swig80():
    # Odczytanie danych z plików CSV za pomocą funkcji 'read_csv_data'
    _, _, swig80_df = read_csv_data()

    if swig80_df is not None:
        # Jeśli ramka danych 'swig80_df' nie jest pusta
        # Wygeneruj szablon HTML 'component.html' z danymi z 'swig80_df'
        return render_template('component.html', data_html=swig80_df.to_html(classes='data', index=False),
                               title='Komponenty sWIG 80')
    else:
        # Jeśli dane nie zostały jeszcze załadowane, wyświetl komunikat o błędzie
        return render_template('component.html', message='Dane nie zostały załadowane. Dokonaj aktualizacji danych.')

if __name__ == '__main__':
        app.run(debug=True)