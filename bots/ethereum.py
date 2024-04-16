import json
import os

import requests
from dotenv import load_dotenv
import pandas as pd
import io
from datetime import datetime
import functools
import time
from bs4 import BeautifulSoup
import re


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"---> {func} Elapsed time: {elapsed_time:0.4f} seconds")
        print("\n")
        return value

    return wrapper_timer


@timer
def get_external_ethereum_blockchain(address, etherscan_api_key):
    url = f"https://api.etherscan.io/api?module=account&amp;action=txlist&amp;address={address}&amp;startblock=0&amp;endblock=99999999&amp;sort=asc&amp;apiKey={etherscan_api_key}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    print(f" get_ethereum_blockchain. url: {url}")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()  # Directamente obtenemos la respuesta en formato dict
            message = data.get("message", "")

            if "NOTOK" in message:
                print(message)  # Esto debería imprimir "NOTOK"

            return data  # Retornamos el JSON ya que el código de estado es 200
        else:
            print(f"Error: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error de conexión: {e}")
        return None
    except json.JSONDecodeError:
        print("Error al decodificar el JSON.")
        return None

@timer
def get_internal_ethereum_blockchain(address, etherscan_api_key):
    url = f"https://api.etherscan.io/api?module=account&amp;action=txlistinternal&amp;address={address}&amp;startblock=0&amp;endblock=99999999&amp;sort=asc&amp;apiKey={etherscan_api_key}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    print(f" get_internal_ethereum_blockchain. url: {url}")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()  # Directamente obtenemos la respuesta en formato dict
            message = data.get("message", "")

            if "NOTOK" in message:
                print(message)  # Esto debería imprimir "NOTOK"

            return data  # Retornamos el JSON ya que el código de estado es 200
        else:
            print(f"Error: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error de conexión: {e}")
        return None
    except json.JSONDecodeError:
        print("Error al decodificar el JSON.")
        return None
@timer
def get_ether_balance(address, api_token):
    # balancemulti or balance
    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={api_token}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    print(f" get_ether_balance. url: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code}")
        return None


@timer
def get_initial_address():
    url = "https://etherscan.io/tx/0x8135f16fb351d1488512859d68de92ec3857f0a0a606466020298e91a69d76f2"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    print(f" get_initial_address. url: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code}")
        return None


def get_eth_balance(address, api_key):
    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={api_key}"
    print(f"get_eth_balance adress: {address} url: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return int(response.json()["result"], 16)
    else:
        print(f"Error: {response.json()['message']}")
        return None


def get_eth_transactions(address, api_key, start_block=14064182, end_block=None):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}"
    if start_block:
        url += f"&startblock={start_block}"
    if end_block:
        url += f"&endblock={end_block}"
    print(
        f"---> get_eth_transactions {address} start_block={start_block} end_block={end_block} url={url}"
    )

    url += f"&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
        # return response.json()["result"]
    else:
        print(f"Error: {response.json()['message']}")
        return None


def get_huobi_transactions(currency, eth_address, start_date, end_date):
    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = "https://api.huobi.pro/v1/query/deposit-withdraw"
    print(
        f"---> get_huobi_transactions currency:{currency} eth_address:{eth_address} start_date:{start_date} end_date:{end_date} url: {url}"
    )
    # Parámetros de la solicitud
    params = {
        "currency": currency,
        "type": "deposit",
        "size": 100,
        "start": start_timestamp * 1000,
        "end": end_timestamp * 1000,
        "address": eth_address,
    }

    try:
        # Realizar la solicitud GET
        response = requests.get(url, params=params)

        # Decode the response content from bytes to string
        response_content = response.content.decode("utf-8")

        # Parse the JSON string
        try:
            response_data = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            response_data = {}

        # Extract the status, err-code, and err-msg fields
        status = response_data.get("status")
        err_code = response_data.get("err-code")
        err_msg = response_data.get("err-msg")

        # Print the extracted fields
        print(f"Status: {status}")
        print(f"Error code: {err_code}")
        print(f"Error message: {err_msg}")

        if response.status_code == 200 and status == "ok":
            return response_data.get("data")
        else:
            print("Error en la solicitud:", err_msg)
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


def get_binance_transactions(currency_pair, eth_address, start_date, end_date):
    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    # Construir la URL de la API
    url = f"https://api.binance.com/api/v3/depositHistory.html?coin={currency_pair}&startTime={start_timestamp}&endTime={end_timestamp}&address={eth_address}"
    print(
        f"---> get_binance_transactions. currency_pair:{currency_pair} eth_address:{eth_address} start_date: {start_date} end_date: {end_date} url: {url}"
    )

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            return data
        else:
            print("Error en la solicitud:", data)
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None
    print("---> get_binance_transactions...")


@timer
def get_ethereum_scam_data(
    address, api_key, parquet_name="transactions_ethereum.parquet"
):
    print(f"---> get_ethereum_scam_data address: {address} parquet_name:{parquet_name}")
    global transactions
    balance = get_eth_balance(address, api_key)
    print(f"Scanning suspicious scamm address: address: {address} balance:{balance}")
    transactions = get_eth_transactions(address, api_key)
    # Cargar datos JSON en un DataFrame de Pandas
    df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
    # Guardar DataFrame como archivo Parquet
    df.to_parquet(parquet_name, index=True)
    print(f"Data saved to {parquet_name}.")
    print("---> get_ethereum_scam_data")


@timer
def get_okex_transactions(currency, eth_address, start_date, end_date):

    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = f"https://www.okex.com/api/account/v3/deposit/history?currency={currency}&start={start_timestamp}&end={end_timestamp}&address={eth_address}"
    print(
        f"---> get_okex_transactions currency:{currency} eth_address: {eth_address} start_date: {start_date} end_date: {end_date} url: {url}"
    )

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            return data
        else:
            print("Error en la solicitud:", data)
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None
    print("---> get_okex_transactions")


@timer
def get_upbit_transactions(currency, eth_address, start_date, end_date):

    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = f"https://api.upbit.com/v1/deposits/ethereum?currency={currency}&access_key={eth_address}&range={start_timestamp}T00:00:00Z_{end_timestamp}T00:00:00Z"
    print(
        f"---> get_upbit_transactions currency: {currency} eth_address: {eth_address} start_date: {start_date} end_date: {end_date} url: {url}"
    )

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            return data
        else:
            print("Error en la solicitud:", data)
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None
    print("---> get_upbit_transactions")


@timer
def get_coinone_transactions(currency, eth_address, start_date, end_date):

    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = f"https://api.coinone.co.kr/deposit/history/?format=json&currency={currency}&access_token={eth_address}&start={start_timestamp}&end={end_timestamp}"
    print(
        f"---> get_coinone_transactions currency: {currency} eth_address:{eth_address} start_date: {start_date} url: {url}"
    )

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            return data
        else:
            print("Error en la solicitud:", data)
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None

    print("---> get_coinone_transactions")


@timer
def get_korbit_transactions(currency_pair, eth_address, start_date, end_date):

    # Formatear las fechas
    start_iso = datetime.strptime(start_date, "%Y-%m-%d").isoformat()
    end_iso = datetime.strptime(end_date, "%Y-%m-%d").isoformat()

    # Construir la URL de la API
    url = f"https://api.korbit.co.kr/v1/user/deposit/{currency_pair}?access_token={eth_address}&start={start_iso}&end={end_iso}"
    print(f"---> get_korbit_transactions currency_pair:{currency_pair} url:{url}")

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            print(f"---> get_korbit_transactions {response.status_code}")
            return data
        else:
            print("Error en la solicitud:", data)
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


@timer
def get_bithumb_transactions(currency, eth_address, start_date, end_date):

    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = f"https://api.bithumb.com/public/transaction_history/{currency}/KRW?searchGb=0&offset=0&count=100&start={start_timestamp}&end={end_timestamp}&address={eth_address}"
    print(f"---> get_bithumb_transactions url:{url}")

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and data["status"] == "0000":
            print(
                f"---> get_bithumb_transactions {response.status_code} {data['status']}"
            )
            return data["data"]
        else:
            print("Error en la solicitud:", data)
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


@timer
def get_bitstamp_transactions(currency_pair, eth_address, start_date, end_date):

    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = f"https://www.bitstamp.net/api/v2/user_transactions/{currency_pair}/?time=hour&start={start_timestamp}&end={end_timestamp}&address={eth_address}"
    print(f"---> get_bitstamp_transactions url:{url}")

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            print(f"---> get_bitstamp_transactions {response.status_code}")

            return data
        else:
            print(f"---> get_bitstamp_transactions {response.status_code}")
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


@timer
def get_coinbase_pro_transactions(product_id, eth_address, start_date, end_date):

    # Formatear las fechas
    start_iso = datetime.strptime(start_date, "%Y-%m-%d").isoformat()
    end_iso = datetime.strptime(end_date, "%Y-%m-%d").isoformat()

    # Construir la URL de la API
    url = f"https://api.pro.coinbase.com/products/{product_id}/trades?start={start_iso}&end={end_iso}&eth_address={eth_address}"
    print(f"---> get_coinbase_pro_transactions url:{url}")

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            print(f"---> get_coinbase_pro_transactions {response.status_code}")
            return data
        else:
            print(f"---> get_coinbase_pro_transactions {response.status_code}")
            return None

    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


@timer
def get_kraken_transactions(pair, eth_address, start_date, end_date):

    # Formatear las fechas
    start_iso = datetime.strptime(start_date, "%Y-%m-%d").isoformat()
    end_iso = datetime.strptime(end_date, "%Y-%m-%d").isoformat()

    # Construir la URL de la API
    url = f"https://api.kraken.com/0/public/Trades?pair={pair}&since={start_iso}&eth_address={eth_address}"
    print(f"---> get_kraken_transactions url:{url}")

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and data["error"] == []:
            print(f"---> get_kraken_transactions url:{url} {response.status_code} ")

            return data["result"]
        else:
            print(
                f"---> get_kraken_transactions url:{url} {response.status_code} {data['error']}"
            )
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


@timer
def get_gemini_transactions(symbol, eth_address, start_date, end_date):

    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = f"https://api.gemini.com/v1/mytrades/{symbol}?limit_trades=1000&timestamp={start_timestamp}&address={eth_address}"
    print(f"---> get_gemini_transactions url: {url}")

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            print(f"---> get_gemini_transactions url: {url} {response.status_code}")

            return data
        else:
            print(f"---> get_gemini_transactions url: {url} {response.status_code}")
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


@timer
def get_kucoin_transactions(symbol, eth_address, start_date, end_date):

    # Formatear las fechas
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Construir la URL de la API
    url = f"https://api.kucoin.com/api/v1/accounts/{symbol}/ledger?startAt={start_timestamp}&endAt={end_timestamp}&address={eth_address}"
    print(f"---> get_kucoin_transactions url:{url}")

    try:
        # Realizar la solicitud GET
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            print(f"---> get_kucoin_transactions url:{url} {response.status_code}")

            return data
        else:
            print(f"---> get_kucoin_transactions url:{url} {response.status_code}")
            return None
    except Exception as e:
        print("Error al procesar la solicitud:", e)
        return None


@timer
def kukoin_transactions():
    global symbol, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    symbol = "ETH"
    eth_address = address_scammer
    transactions = get_kucoin_transactions(symbol, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas get_kucoin_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = (
            f"kucoin_transacctions_{pair}_{eth_address}_{start_date}_{end_date}.parquet"
        )
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones en get_kucoin_transactions")


@timer
def gemini_transactions():
    global symbol, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    symbol = "ethusd"
    eth_address = address_scammer
    transactions = get_gemini_transactions(symbol, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas get_gemini_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = (
            f"gemini_transacctions_{pair}_{eth_address}_{start_date}_{end_date}.parquet"
        )
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones get_gemini_transactions")


@timer
def kraken_transactions():
    global pair, eth_address, transactions, transactions_json, df, parquet_name
    # Ejemplo de uso
    pair = "XETHZUSD"
    eth_address = address_scammer
    transactions = get_kraken_transactions(pair, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas get_kraken_transactions")
        transactions_json = json.dumps(transactions)
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(
            io.BytesIO(transactions_json.encode("utf-8")), encoding="utf-8"
        )

        # Limpiar datos (opcional)
        # ... (código para eliminar nulos o convertir no numéricos)

        # 1. Handle lists in XETHZUSD (assuming the first element is the desired float)
        def handle_list(x):
            # Check if element is a list and extract the first element
            if isinstance(x, list):
                return x[0]
            else:
                return x

        # 2. Apply element-wise transformation using vectorized function
        df["XETHZUSD"] = df["XETHZUSD"].apply(handle_list)

        # 3. Convert to float (assuming the first element is the desired float)
        df["XETHZUSD"] = df["XETHZUSD"].astype(float)

        # Verificar tipo de datos
        print(df["XETHZUSD"].dtype)
        parquet_name = (
            f"kraken_transacctions_{pair}_{eth_address}_{start_date}_{end_date}.parquet"
        )
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones get_kraken_transactions")


@timer
def coinbase_pro_transactions():
    global eth_address, transactions, transactions_json, df, parquet_name
    # Ejemplo de uso
    product_id = "ETH-USD"
    eth_address = address_scammer
    transactions = get_coinbase_pro_transactions(
        product_id, eth_address, start_date, end_date
    )
    if transactions:
        print("Transacciones encontradas coinbase_pro")
        # Convertir la lista de transacciones a un objeto JSON
        transactions_json = json.dumps(transactions)
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(
            io.BytesIO(transactions_json.encode("utf-8")), encoding="utf-8"
        )
        parquet_name = f"coinbase_pro_transacctions_{product_id}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_coinbase_pro_transactions")


@timer
def bitstamp_transactions():
    global currency_pair, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    currency_pair = "ethusd"
    eth_address = address_scammer
    transactions = get_bitstamp_transactions(
        currency_pair, eth_address, start_date, end_date
    )
    if transactions:
        print("Transacciones encontradas:", transactions)
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"bitstamp_transacctions_{currency_pair}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_bitstamp_transactions")


@timer
def bithump_transactions():
    global currency, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    currency = "ETH"
    eth_address = address_scammer
    transactions = get_bithumb_transactions(currency, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas: get_bithumb_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"bithumb_transacctions_{currency}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_bithumb_transactions")


@timer
def korbit_transactions():
    global currency_pair, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    currency_pair = "eth_krw"
    eth_address = address_scammer
    transactions = get_korbit_transactions(
        currency_pair, eth_address, start_date, end_date
    )
    if transactions:
        print("Transacciones encontradas: get_korbit_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"korbit_transacctions_{currency_pair}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_korbit_transactions")


@timer
def coinone_transactions():
    global currency, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    currency = "eth"
    eth_address = address_scammer
    transactions = get_coinone_transactions(currency, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas: get_coinone_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"coinone_transacctions_{currency}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_coinone_transactions")


@timer
def upbit_transactions():
    global currency, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    currency = "ETH"
    eth_address = address_scammer
    transactions = get_upbit_transactions(currency, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas: get_upbit_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"upbit_transacctions_{currency}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_upbit_transactions")


@timer
def okex_transactions():
    global currency, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    currency = "eth"
    eth_address = address_scammer
    transactions = get_okex_transactions(currency, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas: get_okex_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"okex_transacctions_{currency}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_okex_transactions")


@timer
def huobi_transactions():
    global currency, eth_address, transactions, df, parquet_name
    # Ejemplo de uso
    currency = "eth"
    eth_address = address_scammer
    transactions = get_huobi_transactions(currency, eth_address, start_date, end_date)
    if transactions:
        print("Transacciones encontradas: get_huobi_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"huobi_transacctions_{currency}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_huobi_transactions")


@timer
def binance_transactions():
    global currency_pair, eth_address, start_date, end_date, transactions, df, parquet_name
    # Ejemplo de uso
    currency_pair = "ETH"
    eth_address = address_scammer
    start_date = "2021-01-01"
    end_date = "2024-04-01"
    transactions = get_binance_transactions(
        currency_pair, eth_address, start_date, end_date
    )
    if transactions:
        print("Transacciones encontradas: get_binance_transactions")
        # Cargar datos JSON en un DataFrame de Pandas
        df = pd.read_json(io.BytesIO(transactions), encoding="utf-8")
        parquet_name = f"binance_transacctions_{currency_pair}_{eth_address}_{start_date}_{end_date}.parquet"
        # Guardar DataFrame como archivo Parquet
        df.to_parquet(parquet_name, index=True)
        print(f"Data saved to {parquet_name}.")
    else:
        print("No se encontraron transacciones. get_binance_transactions")


def wrapping_html_content(html):
    # Supongamos que `html_content` es tu cadena HTML
    html_content = html

    # Convertir bytes a string si es necesario
    if isinstance(html_content, bytes):
        html_content = html_content.decode("utf-8")

    soup = BeautifulSoup(html_content, "html.parser")

    # Diccionario para almacenar los valores extraídos
    extracted_info = {}

    # Buscar y extraer Transaction Hash
    transaction_hash_span = soup.find("span", id="spanTxHash")
    if transaction_hash_span:
        extracted_info["Transaction Hash"] = transaction_hash_span.text

    # Buscar y extraer Status
    status_span = soup.find("span", text=re.compile(r"Status:"))
    if status_span:
        # El status real está en el siguiente elemento span con una clase específica
        status = status_span.find_next("span")
        if status:
            extracted_info["Status"] = status.get_text()

    # Buscar y extraer Block
    block_a_tag = soup.find("a", href=re.compile(r"/block/"))
    if block_a_tag:
        extracted_info["Block"] = block_a_tag.text

    # Buscar y extraer Timestamp
    timestamp_span = soup.find("span", id="showUtcLocalDate")
    if timestamp_span:
        extracted_info["Timestamp"] = timestamp_span.text

    # Buscar y extraer Transaction Action
    transaction_action_div = soup.find("div", id="wrapperContent")
    if transaction_action_div:
        transaction_action = transaction_action_div.get_text(strip=True, separator=" ")
        extracted_info["Transaction Action"] = " ".join(transaction_action.split())

    # Buscar y extraer From
    from_a_tag = soup.find(
        "a", text=re.compile(r"From:"), attrs={"data-highlight-target": True}
    )
    if from_a_tag:
        extracted_info["From"] = from_a_tag.text

    # Buscar y extraer Interacted With (To)
    interacted_with_span = soup.find(
        "span", text=re.compile(r"Interacted With \(To\):")
    )
    if interacted_with_span:
        # El valor real está en el siguiente elemento a
        interacted_with = interacted_with_span.find_next(
            "a", {"data-highlight-value": True}
        )
        if interacted_with:
            extracted_info["Interacted With (To)"] = interacted_with.get_text()

    # Imprimir los resultados
    for key, value in extracted_info.items():
        print(f"{key}: {value}")

@timer
def get_full_address_from_etherscan(my_address, etherscan_api_key):
    global transactions, data
    json_external_etherscan = get_external_ethereum_blockchain(my_address, etherscan_api_key)
    from_external_addresses = set()
    to_external_addresses = set()
    # Crear un conjunto para guardar direcciones únicas
    direcciones_internal = set()
    # Asegurate que la respuesta contiene las transacciones
    if json_external_etherscan['status'] == '1' and 'result' in json_external_etherscan:
        transactions = json_external_etherscan['result']

        # Procesa cada transacción para extraer las direcciones 'from' y 'to'
        for tx in transactions:
            from_external_addresses.add(tx['from'])
            to_external_addresses.add(tx['to'])

        # En este punto, `from_addresses` y `to_addresses` contienen todas las direcciones únicas que buscamos
        print(f"From addresses: {from_external_addresses}")
        print(f"To addresses: {to_external_addresses}")

        # Si necesitas trabajar con estos como listas en vez de conjuntos
        from_address_list = list(from_external_addresses)
        to_address_list = list(to_external_addresses)

    else:
        print("No se encontraron transacciones o hubo un error en la petición.")
    print(f"{json_external_etherscan}")
    json_internal_etherscan = get_internal_ethereum_blockchain(my_address, etherscan_api_key)
    if json_internal_etherscan['status'] == '1' and 'result' in json_internal_etherscan:
        data = json_internal_etherscan.json()

        # Recorrer las transacciones y añadir las direcciones "from" y "to"
        for tx in data["result"]:
            direcciones_internal.add(tx["from"])
            direcciones_internal.add(tx["to"])
    else:
        print("No se encontraron transacciones o hubo un error en la petición.")
    print(f"{json_internal_etherscan}")
    total_addresses = from_external_addresses | to_external_addresses | direcciones_internal
    return total_addresses

if __name__ == "__main__":
    load_dotenv()
    # Direccion a donde fueron mis fondos
    my_address = "0x46C218cb1C09697840A6eAe78A54821cFdD9fD04"
    address_scammer = "0xaca6c427e543240a74f9438cd3e8769b144c4c55"
    etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")

    full_set_adress = get_full_address_from_etherscan(my_address, etherscan_api_key)
    print(f"full_set_adress: {full_set_adress}")
    # html_content = get_initial_address()
    # wrapping_html_content(html_content)

    # retrieve data from etherscan and saved to parquet file
    # get_ethereum_scam_data(address_scammer, etherscan_api_key)

    # binance_transactions()

    # huobi_transactions()

    # okex_transactions()

    # upbit_transactions()

    # coinone_transactions()

    # korbit_transactions()

    # bithump_transactions()

    # bitstamp_transactions()

    # coinbase_pro_transactions()

    # kraken_transactions()

    # gemini_transactions()

    # kukoin_transactions()
