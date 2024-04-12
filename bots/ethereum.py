import os

import requests
from dotenv import load_dotenv


def get_eth_balance(address, api_key):
    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return int(response.json()["result"], 16)
    else:
        print(f"Error: {response.json()['message']}")
        return None


def get_eth_transactions(address, api_key, start_block=None, end_block=None):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}"
    if start_block:
        url += f"&startblock={start_block}"
    if end_block:
        url += f"&endblock={end_block}"
    url += f"&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
        # return response.json()["result"]
    else:
        print(f"Error: {response.json()['message']}")
        return None


if __name__ == "__main__":
    load_dotenv()
    address = "0xaca6c427e543240a74f9438cd3e8769b144c4c55"
    api_key=os.getenv("ETHERSCAN_API_KEY")
    balance = get_eth_balance(address, api_key)
    print(f"Escaneando dirección: {address}...")
    print(f"Balance: {balance} wei")
    transactions = get_eth_transactions(address, api_key)
    for tx in transactions:
        print(tx)
